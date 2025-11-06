//! Implements the semantic analysis stage of the frontend

allocator: std.mem.Allocator,
scope_stack: std.ArrayList(struct {
    identifiers: token_map.Map(IdentifierDefinition) = .{},
}) = .{},
procedures: token_map.Map(Procedure) = .{},
types: std.MultiArrayList(Type) = .{},

air_builder: struct {
    instructions: std.ArrayList(spirv.Air.Instruction) = .{},
    blocks: std.ArrayList(spirv.Air.Block) = .{},
    functions: std.ArrayList(spirv.Air.Function) = .{},
    variables: std.ArrayList(spirv.Air.Variable) = .{},
    types: std.ArrayList(spirv.Air.Type) = .{},
} = .{},
errors: std.ArrayList(Ast.Error) = .{},

pub const Procedure = struct {
    return_type: TypeIndex,
    parameters: []const Parameter,

    pub const Parameter = struct {
        type_index: TypeIndex,
        qualifier: Qualifier,

        pub const Qualifier = enum {
            constant,
            in,
            inout,
            out,
        };
    };
};

pub const IdentifierDefinition = struct {
    token_index: Ast.TokenIndex,
    type_index: TypeIndex,
};

pub fn deinit(
    self: *Sema,
    allocator: std.mem.Allocator,
) void {
    for (self.scope_stack.items) |*scope| {
        scope.identifiers.deinit(allocator);
    }

    self.scope_stack.deinit(allocator);
    self.types.deinit(allocator);
    self.errors.deinit(allocator);

    self.* = undefined;
}

pub const Type = struct {
    tag: Tag,
    data_start: u32,
    data_end: u32,

    pub const Tag = enum {
        literal_int,
        literal_float,
        literal_string,
        bool,
        int,
        uint,
        float,
        double,
        vec2,
        vec3,
        vec4,
        @"struct",
    };
};

///Analyse the root node of the ast
pub fn analyse(ast: Ast, allocator: std.mem.Allocator) !struct {
    spirv.Air,
    []Ast.Error,
} {
    const root_decls = ast.root_decls;

    var sema: Sema = .{
        .allocator = allocator,
    };
    defer sema.deinit(allocator);

    try sema.scopePush();
    defer sema.scopePop();

    for (root_decls) |decl| {
        const node_tag = decl.tag;

        switch (node_tag) {
            .procedure => {
                sema.analyseProcedure(ast, decl) catch |e| {
                    switch (e) {
                        error.IdentifierAlreadyDefined,
                        error.TypeMismatch,
                        error.UndeclaredIdentifier,
                        => {},
                        else => return e,
                    }
                };
            },
            else => {},
        }
    }

    return .{
        .{
            .capability = .Shader,
            .addressing_mode = .logical,
            .memory_model = .vulkan,
            .entry_point = .{
                .execution_mode = .vertex,
                .name = "main",
                .interface = &.{},
            },
            .instructions = &.{},
            .blocks = &.{},
            .functions = &.{},
            .variables = &.{},
            .types = &.{},
        },
        try sema.errors.toOwnedSlice(allocator),
    };
}

pub fn analyseProcedure(
    self: *Sema,
    ast: Ast,
    node: Ast.NodeIndex,
) !void {
    try self.scopePush();
    defer self.scopePop();

    const procedure: Ast.Node.Procedure = ast.dataFromNode(node, .procedure);
    const param_list: Ast.Node.ParamList = ast.dataFromNode(procedure.param_list, .param_list);

    const procedure_definition_get_result = try self.procedures.getOrPut(self.allocator, ast.tokenString(procedure.name));
    const procedure_definition = procedure_definition_get_result.value_ptr;

    if (!procedure_definition_get_result.found_existing) {
        procedure_definition.* = .{
            .parameters = &.{},
            .return_type = .void,
        };
    }

    procedure_definition.return_type = try self.resolveTypeFromTypeExpr(ast, procedure.return_type);

    if (!procedure.param_list.isNil()) {
        const parameters = try self.allocator.alloc(Procedure.Parameter, param_list.params.len);
        errdefer self.allocator.free(parameters);

        procedure_definition.parameters = parameters;

        for (param_list.params, 0..) |param_node, param_index| {
            const param: Ast.Node.ParamExpr = ast.dataFromNode(param_node, .param_expr);
            const param_definition = &parameters[param_index];

            const param_type = try self.resolveTypeFromTypeExpr(ast, param.type_expr);

            param_definition.qualifier = switch (param.qualifier) {
                .keyword_inout => .inout,
                .keyword_out => .out,
                .keyword_in => .in,
                .keyword_const => .constant,
                else => unreachable,
            };

            param_definition.type_index = param_type;

            try self.scopeDefine(ast, param.name, param_type);
        }
    }

    //TODO: handle forward declaration
    if (procedure.body.isNil()) return;

    const body = ast.dataFromNode(procedure.body, .statement_block);

    for (body.statements) |statement_node| {
        switch (statement_node.tag) {
            .statement_var_init => {
                const var_init: Ast.Node.StatementVarInit = ast.dataFromNode(statement_node, .statement_var_init);

                const type_index = try self.resolveTypeFromExpression(ast, statement_node);

                try self.scopeDefine(ast, var_init.identifier, type_index);
            },
            else => {},
        }
    }
}

///Handles function overloading
pub fn resolveProcedure(
    self: *Sema,
    ast: Ast,
    name: Ast.NodeIndex,
    param_or_arg_list: Ast.NodeIndex,
) !*Procedure {
    _ = param_or_arg_list; // autofix

    const name_data: Ast.Node.Identifier = ast.dataFromNode(name, .expression_identifier);

    //TODO: handle procedure overloading
    const procedure_definition = self.procedures.getPtr(ast.tokenString(name_data.token)) orelse {
        try self.errors.append(self.allocator, .{
            .tag = .undeclared_identifier,
            .token = name_data.token,
        });

        return error.UndeclaredIdentifier;
    };

    return procedure_definition;
}

pub fn scopePush(self: *Sema) !void {
    try self.scope_stack.append(self.allocator, .{});
}

pub fn scopePop(self: *Sema) void {
    if (self.scope_stack.items.len == 0) {
        return;
    }

    const scope = &self.scope_stack.items[self.scope_stack.items.len - 1];

    scope.identifiers.deinit(self.allocator);

    self.scope_stack.items.len = 0;
}

pub fn scopeDefine(
    self: *Sema,
    ast: Ast,
    identifier_token: Ast.TokenIndex,
    type_index: TypeIndex,
) !void {
    const scope = &self.scope_stack.items[self.scope_stack.items.len - 1];

    const identifier_string = ast.tokenString(identifier_token);

    if (scope.identifiers.get(identifier_string)) |original_definition| {
        try self.errors.append(self.allocator, .{
            .tag = .identifier_redefined,
            .token = identifier_token,
            .data = .{
                .identifier_redefined = .{
                    .redefinition_identifier = identifier_token,
                    .definition_identifier = original_definition.token_index,
                },
            },
        });

        return error.IdentifierAlreadyDefined;
    }

    try scope.identifiers.put(self.allocator, identifier_string, .{
        .type_index = type_index,
        .token_index = identifier_token,
    });
}

pub fn scopeResolve(self: *Sema, ast: Ast, identifier_token: Ast.TokenIndex) !*IdentifierDefinition {
    const identifier_string = ast.tokenString(identifier_token);

    for (1..self.scope_stack.items.len + 1) |reverse_index| {
        const index = self.scope_stack.items.len - reverse_index;
        const scope = &self.scope_stack.items[index];

        if (scope.identifiers.getPtr(identifier_string)) |definition| {
            return definition;
        }
    }

    try self.errors.append(self.allocator, .{
        .tag = .undeclared_identifier,
        .token = identifier_token,
    });

    return error.UndeclaredIdentifier;
}

pub fn resolveTypeFromTypeExpr(self: Sema, ast: Ast, type_expr: Ast.NodeIndex) !TypeIndex {
    _ = self; // autofix

    const type_expr_data = ast.dataFromNode(type_expr, .type_expr);

    const type_expr_token = type_expr_data.token;

    const token_tag: Token.Tag = ast.tokens.items(.tag)[type_expr_token];

    return switch (token_tag) {
        .keyword_uint => .uint,
        .keyword_int => .int,
        .keyword_float => .float,
        .keyword_bool => .bool,
        .keyword_void => .void,
        else => unreachable,
    };
}

pub fn resolveTypeFromExpression(self: *Sema, ast: Ast, expression: Ast.NodeIndex) !TypeIndex {
    switch (expression.tag) {
        .expression_literal_boolean => return .bool,
        .expression_literal_number => {
            const number_literal_data: Ast.Node.ExpressionLiteralNumber = ast.dataFromNode(expression, .expression_literal_number);

            //TODO: optimize this by using a custom parse function which just determines type
            if (std.fmt.parseInt(u64, ast.tokenString(number_literal_data.token), 10)) |_| {
                return .literal_uint;
            } else |e| {
                switch (e) {
                    else => {},
                }
            }

            if (std.fmt.parseInt(i64, ast.tokenString(number_literal_data.token), 10)) |_| {
                return .literal_int;
            } else |e| {
                switch (e) {
                    else => {},
                }
            }

            if (std.fmt.parseFloat(f64, ast.tokenString(number_literal_data.token))) |_| {
                return .literal_float;
            } else |e| {
                switch (e) {
                    else => {},
                }
            }

            unreachable;
        },
        .statement_var_init => {
            const var_init: Ast.Node.StatementVarInit = ast.dataFromNode(expression, .statement_var_init);

            const type_index = try self.resolveTypeFromTypeExpr(ast, var_init.type_expr);
            const expression_type_index = try self.resolveTypeFromExpression(ast, var_init.expression);

            const resultant_type = self.coerceTypeAssign(ast, type_index, expression_type_index) orelse {
                try self.errors.append(self.allocator, .{
                    .tag = .type_mismatch,
                    //identifier + 1 points to the = symbol
                    .token = var_init.identifier + 1,
                    .data = .{
                        .type_mismatch = .{
                            .lhs_type = type_index,
                            .rhs_type = expression_type_index,
                        },
                    },
                });

                return error.TypeMismatch;
            };

            return resultant_type;
        },
        .expression_binary_eql,
        .expression_binary_neql,
        .expression_binary_leql,
        .expression_binary_geql,
        => return .bool,
        .expression_binary_add,
        .expression_binary_sub,
        .expression_binary_mul,
        .expression_binary_div,
        => {
            const binary_expr: Ast.Node.BinaryExpression = ast.dataFromNode(expression, .expression_binary_add);

            const lhs_type = try self.resolveTypeFromExpression(ast, binary_expr.left);
            const rhs_type = try self.resolveTypeFromExpression(ast, binary_expr.right);

            const resultant_type = self.coerceTypeBinaryOp(ast, lhs_type, rhs_type) orelse {
                try self.errors.append(self.allocator, .{
                    .tag = .type_mismatch,
                    .token = binary_expr.op_token,
                    .data = .{
                        .type_mismatch = .{
                            .lhs_type = lhs_type,
                            .rhs_type = rhs_type,
                        },
                    },
                });

                return error.TypeMismatch;
            };

            return resultant_type;
        },
        .expression_identifier => {
            const identifier: Ast.Node.Identifier = ast.dataFromNode(expression, .expression_identifier);

            const definition = try self.scopeResolve(ast, identifier.token);

            return definition.type_index;
        },
        .expression_binary_proc_call => {
            const binary_expr: Ast.Node.BinaryExpression = ast.dataFromNode(expression, .expression_binary_proc_call);

            const identifier = binary_expr.left;
            const arg_list = binary_expr.right;

            const procedure = try self.resolveProcedure(ast, identifier, arg_list);

            return procedure.return_type;
        },
        else => unreachable,
    }
}

pub fn coerceTypeAssign(self: Sema, ast: Ast, lhs: TypeIndex, rhs: TypeIndex) ?TypeIndex {
    _ = self; // autofix
    _ = ast; // autofix

    if (lhs == rhs) {
        return lhs;
    }

    switch (lhs) {
        .int => {
            if (rhs == .literal_int or rhs == .literal_uint) {
                return .int;
            }
        },
        .uint => {
            if (rhs == .literal_uint) {
                return .uint;
            }
        },
        .float => {
            if (rhs == .literal_int or rhs == .literal_uint) {
                return .float;
            }

            if (rhs == .literal_float) {
                return .float;
            }
        },
        else => {},
    }

    return null;
}

pub fn coerceTypeBinaryOp(self: Sema, ast: Ast, lhs: TypeIndex, rhs: TypeIndex) ?TypeIndex {
    _ = self; // autofix
    _ = ast; // autofix

    if (lhs == rhs) {
        return lhs;
    }

    switch (lhs) {
        .int => {
            switch (rhs) {
                .literal_int,
                .literal_uint,
                .uint,
                => return .int,
                .literal_float,
                .float,
                => return .float,
                else => {},
            }
        },
        .literal_int => {
            switch (rhs) {
                .literal_uint,
                => return .literal_int,
                .int,
                .uint,
                => return .int,
                .literal_float,
                => return .literal_float,
                .float => return .float,
                else => {},
            }
        },
        .uint => {
            switch (rhs) {
                .literal_uint,
                => return .uint,
                .literal_float,
                .float,
                => return .float,
                else => {},
            }
        },
        .literal_uint => {
            switch (rhs) {
                .literal_int,
                => return .literal_int,
                .int,
                => return .int,
                .uint => return .uint,
                .literal_float,
                => return .literal_float,
                .float => return .float,
                else => {},
            }
        },
        .float => {
            switch (rhs) {
                .literal_int,
                .literal_uint,
                .literal_float,
                => {
                    return .float;
                },
                else => {},
            }
        },
        else => {},
    }

    return null;
}

pub fn typeName(self: Sema, type_index: TypeIndex) []const u8 {
    _ = self; // autofix
    if (type_index.toArrayIndex() == null) {
        return @tagName(type_index);
    }

    unreachable;
}

pub const TypeIndex = enum(u32) {
    null = 0,
    void,
    uint,
    int,
    float,
    bool,

    literal_int,
    literal_uint,
    literal_float,

    _,

    pub fn toArrayIndex(type_index: TypeIndex) ?usize {
        const integer = @intFromEnum(type_index);
        const offset = @intFromEnum(TypeIndex.bool) + 1;

        if (integer < offset) {
            return null;
        }

        return integer - offset;
    }
};

const std = @import("std");
const Ast = @import("Ast.zig");
const spirv = @import("../spirv.zig");
const Sema = @This();
const token_map = @import("token_map.zig");
const Token = @import("Tokenizer.zig").Token;
