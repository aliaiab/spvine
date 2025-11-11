//! Implements the semantic analysis stage of the frontend

gpa: std.mem.Allocator,
scope_stack: std.ArrayList(struct {
    identifiers: token_map.Map(IdentifierDefinition) = .{},
}) = .{},
procedures: std.ArrayList(Procedure) = .{},
procedure_overloads: token_map.Map(ProcedureOverloads) = .{},
types: std.ArrayList(Type) = .{},
type_map: token_map.Map(TypeIndex) = .{},

air_builder: struct {
    instructions: std.ArrayList(spirv.Air.Instruction) = .{},
    blocks: std.ArrayList(spirv.Air.Block) = .{},
    functions: std.ArrayList(spirv.Air.Function) = .{},
    variables: std.ArrayList(spirv.Air.Variable) = .{},
    types: std.ArrayList(spirv.Air.Type) = .{},
} = .{},
errors: std.ArrayList(Ast.Error) = .{},

///Points to a specific overload of a wider function name
pub const ProcedureIndex = enum(u32) { _ };

pub const ProcedureOverloads = struct {
    return_type: TypeIndex,
    parameter_qualifiers: []const ValueQualifier,
    //The range of parameter counts across overloads
    min_parameter_count: u32,
    max_parameter_count: u32,
    overloads: std.ArrayHashMapUnmanaged([]const TypeIndex, ProcedureIndex, TypeSignatureMapContext, true),
};

pub const Procedure = struct {
    general_data: *ProcedureOverloads,
    parameter_types: []const TypeIndex,
    body_defined: bool,
};

pub const TypeSignatureMapContext = struct {
    pub fn eql(_: @This(), a: []const TypeIndex, b: []const TypeIndex, b_index: usize) bool {
        _ = b_index; // autofix
        if (a.len != b.len) return false;

        for (a, b) |type_a, type_b| {
            if (type_a != type_b) {
                return false;
            }
        }

        return true;
    }

    pub fn hash(_: @This(), a: []const TypeIndex) u32 {
        const a_as_bytes: []const u8 = @ptrCast(a);

        return std.hash.XxHash32.hash(0, a_as_bytes);
    }
};

pub const IdentifierDefinition = struct {
    token_index: Ast.TokenIndex,
    type_index: TypeIndex,
    qualifier: ValueQualifier,
    initial_value: ?u64,
};

pub const ValueQualifier = enum {
    constant,
    in,
    inout,
    out,
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

pub const Type = union(enum) {
    @"struct": struct {
        name: Ast.TokenIndex,
        fields: token_map.Map(StructField),
    },

    pub const StructField = struct {
        type_index: TypeIndex,
        node: Ast.NodeIndex,
    };
};

///Analyse the root node of the ast
pub fn analyse(sema: *Sema, ast: Ast, allocator: std.mem.Allocator) !struct {
    spirv.Air,
    []Ast.Error,
} {
    const root_decls = ast.root_decls;

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
                        error.TypeIncompatibilty,
                        error.ModifiedConstant,
                        error.ArgumentCountMismatch,
                        error.NoMatchingOverload,
                        error.CannotPeformFieldAccess,
                        error.ArrayIndexOutOfBounds,
                        error.ExpressionNotIndexable,
                        error.ExpectedConstantExpression,
                        error.NoFieldInStruct,
                        => {},
                        else => return e,
                    }
                };
            },
            .struct_definition => {
                sema.analyseStructDefinition(ast, decl) catch |e| {
                    switch (e) {
                        error.IdentifierAlreadyDefined => {},
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

pub fn analyseStructDefinition(
    self: *Sema,
    ast: Ast,
    node: Ast.NodeIndex,
) !void {
    const definition: Ast.Node.StructDefinition = ast.dataFromNode(node, .struct_definition);

    var fields: token_map.Map(Type.StructField) = .{};

    for (definition.fields) |field_node| {
        const field_node_data: Ast.Node.StructField = ast.dataFromNode(field_node, .struct_field);

        const type_expr = try self.resolveTypeFromTypeExpr(ast, field_node_data.type_expr);

        if (fields.get(ast.tokenString(field_node_data.name))) |original_field| {
            const original_field_node_data: Ast.Node.StructField = ast.dataFromNode(original_field.node, .struct_field);

            try self.errors.append(self.gpa, .{
                .tag = .identifier_redefined,
                .anchor = .{ .token = field_node_data.name },
                .data = .{
                    .identifier_redefined = .{
                        .redefinition_identifier = field_node_data.name,
                        .definition_identifier = original_field_node_data.name,
                    },
                },
            });

            return error.IdentifierAlreadyDefined;
        }

        try fields.putNoClobber(
            self.gpa,
            ast.tokenString(field_node_data.name),
            .{ .type_index = type_expr, .node = field_node },
        );
    }

    if (self.type_map.get(ast.tokenString(definition.name))) |type_index| {
        const type_data = self.types.items[type_index.toArrayIndex().?];

        try self.errors.append(self.gpa, .{
            .tag = .identifier_redefined,
            .anchor = .{ .token = definition.name },
            .data = .{
                .identifier_redefined = .{
                    .redefinition_identifier = definition.name,
                    .definition_identifier = type_data.@"struct".name,
                },
            },
        });

        return error.IdentifierAlreadyDefined;
    }

    const type_index: TypeIndex = .fromArrayIndex(self.types.items.len);

    try self.types.append(self.gpa, .{
        .@"struct" = .{
            .name = definition.name,
            .fields = fields,
        },
    });

    try self.type_map.put(
        self.gpa,
        ast.tokenString(definition.name),
        type_index,
    );
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

    const procedure_definition_get_result = try self.procedure_overloads.getOrPut(self.gpa, ast.tokenString(procedure.name));
    const procedure_definition = procedure_definition_get_result.value_ptr;

    if (!procedure_definition_get_result.found_existing) {
        procedure_definition.* = .{
            .overloads = .{},
            .min_parameter_count = @as(u32, @intCast(param_list.params.len)),
            .max_parameter_count = @as(u32, @intCast(param_list.params.len)),
            .parameter_qualifiers = &.{},
            .return_type = .void,
        };
    }

    procedure_definition.return_type = try self.resolveTypeFromTypeExpr(ast, procedure.return_type);

    var procedure_index: ProcedureIndex = undefined;

    //TODO: use an arena
    const parameter_types = try self.gpa.alloc(TypeIndex, param_list.params.len);

    const parameter_qualifiers = try self.gpa.alloc(ValueQualifier, param_list.params.len);

    for (param_list.params, 0..) |param_node, param_index| {
        const param: Ast.Node.ParamExpr = ast.dataFromNode(param_node, .param_expr);
        const param_definition_type = &parameter_types[param_index];
        const param_definition_qualifier = &parameter_qualifiers[param_index];

        const param_type = try self.resolveTypeFromTypeExpr(ast, param.type_expr);

        param_definition_qualifier.* = switch (param.qualifier) {
            .keyword_inout => .inout,
            .keyword_out => .out,
            .keyword_in => .in,
            .keyword_const => .constant,
            else => unreachable,
        };

        param_definition_type.* = param_type;

        try self.scopeDefine(
            ast,
            param.name,
            param_type,
            param_definition_qualifier.*,
            null,
        );
    }

    if (!procedure_definition_get_result.found_existing) {
        procedure_definition_get_result.value_ptr.parameter_qualifiers = parameter_qualifiers;
    } else {
        //TODO: emit an error if qualifiers don't match
    }

    const overload_query = try procedure_definition.overloads.getOrPut(
        self.gpa,
        parameter_types,
    );

    procedure_definition.max_parameter_count = @max(procedure_definition.max_parameter_count, @as(u32, @intCast(param_list.params.len)));
    procedure_definition.min_parameter_count = @min(procedure_definition.min_parameter_count, @as(u32, @intCast(param_list.params.len)));

    if (overload_query.found_existing) {
        procedure_index = overload_query.value_ptr.*;
        const procedure_overload = self.procedures.items[@intFromEnum(procedure_index)];

        if (procedure_overload.body_defined or procedure.body == Ast.NodeIndex.nil) {
            try self.errors.append(self.gpa, .{
                .tag = .identifier_redefined,
                .anchor = .{ .token = procedure.name },
                .data = .{
                    .identifier_redefined = .{
                        .redefinition_identifier = procedure.name,
                        .definition_identifier = procedure.name,
                    },
                },
            });

            return error.IdentifierAlreadyDefined;
        }
    } else {
        procedure_index = @enumFromInt(@as(u32, @intCast(self.procedures.items.len)));
        const procedure_overload = try self.procedures.addOne(self.gpa);

        procedure_overload.parameter_types = parameter_types;
        procedure_overload.general_data = procedure_definition;
        procedure_overload.body_defined = procedure.body != Ast.NodeIndex.nil;

        overload_query.value_ptr.* = procedure_index;
    }

    if (procedure.body == Ast.NodeIndex.nil) return;

    const procedure_data = &self.procedures.items[@intFromEnum(procedure_index)];

    try self.analyseStatement(
        ast,
        procedure_data,
        procedure.body,
        .{ .block_new_scope = false },
    );
}

pub fn analyseStatement(
    self: *Sema,
    ast: Ast,
    procedure: *const Procedure,
    statement_node: Ast.NodeIndex,
    options: struct {
        ///Whether a statement_block should cause a new scope
        block_new_scope: bool = true,
    },
) !void {
    if (statement_node == Ast.NodeIndex.nil) {
        return;
    }

    switch (statement_node.tag) {
        .statement_var_init => {
            const var_init: Ast.Node.StatementVarInit = ast.dataFromNode(statement_node, .statement_var_init);

            var type_index = try self.resolveTypeFromTypeExpr(ast, var_init.type_expr);

            if (var_init.array_length_specifier != Ast.NodeIndex.nil) {
                const array_length = self.resolveConstantExpression(ast, var_init.array_length_specifier) catch |e| {
                    switch (e) {
                        error.ConstantEvaluationFailed => {
                            try self.errors.append(self.gpa, .{
                                .anchor = .{ .node = var_init.array_length_specifier },
                                .tag = .expected_constant_expression,
                            });

                            return error.ExpectedConstantExpression;
                        },
                        else => return e,
                    }
                };

                type_index = .array(type_index, @intCast(array_length));
            }

            var initial_value: ?u64 = null;

            if (var_init.expression != Ast.NodeIndex.nil) {
                const expression_type_index = try self.resolveTypeFromExpression(ast, var_init.expression);

                _ = switch (coerceTypeAssign(type_index, expression_type_index)) {
                    .null => {
                        try self.errors.append(self.gpa, .{
                            .tag = .type_mismatch,
                            .anchor = .{ .node = var_init.expression },
                            .data = .{
                                .type_mismatch = .{
                                    .lhs_type = type_index,
                                    .rhs_type = expression_type_index,
                                },
                            },
                        });

                        return error.TypeMismatch;
                    },
                    else => |res| res,
                };

                if (var_init.qualifier == .keyword_const and expression_type_index.toData().literal == 1) {
                    initial_value = try self.resolveConstantExpression(ast, var_init.expression);
                }
            }

            try self.scopeDefine(
                ast,
                var_init.identifier,
                type_index,
                switch (var_init.qualifier) {
                    .keyword_in => .in,
                    .keyword_const => .constant,
                    else => unreachable,
                },
                initial_value,
            );
        },
        .statement_if => {
            const statement_if: Ast.Node.StatementIf = ast.dataFromNode(statement_node, .statement_if);

            const condition_type = try self.resolveTypeFromExpression(ast, statement_if.condition_expression);

            _ = switch (coerceType(.bool, condition_type)) {
                .null => {
                    try self.errors.append(self.gpa, .{
                        .tag = .type_mismatch,
                        .anchor = .{ .node = statement_if.condition_expression },
                        .data = .{
                            .type_mismatch = .{
                                .lhs_type = .bool,
                                .rhs_type = condition_type,
                            },
                        },
                    });

                    return error.TypeMismatch;
                },
                else => |res| res,
            };

            try self.analyseStatement(ast, procedure, statement_if.taken_statement, .{});
            try self.analyseStatement(ast, procedure, statement_if.not_taken_statement, .{});
        },
        .statement_block => {
            const body = ast.dataFromNode(statement_node, .statement_block);

            if (options.block_new_scope) try self.scopePush();
            defer if (options.block_new_scope) self.scopePop();

            for (body.statements) |sub_statement| {
                try self.analyseStatement(ast, procedure, sub_statement, .{});
            }
        },
        .expression_binary_assign,
        .expression_binary_assign_add,
        .expression_binary_assign_sub,
        .expression_binary_assign_mul,
        .expression_binary_assign_div,
        .expression_binary_assign_bitwise_shift_left,
        .expression_binary_assign_bitwise_shift_right,
        => {
            _ = try self.resolveTypeFromExpression(ast, statement_node);
        },
        .statement_return => {
            const statement_return: Ast.Node.StatementReturn = ast.dataFromNode(statement_node, .statement_return);

            const expr_type = if (statement_return.expression != Ast.NodeIndex.nil) try self.resolveTypeFromExpression(ast, statement_return.expression) else .void;

            _ = switch (coerceTypeAssign(procedure.general_data.return_type, expr_type)) {
                .null => {
                    try self.errors.append(self.gpa, .{
                        .tag = .type_mismatch,
                        .anchor = if (statement_return.expression != Ast.NodeIndex.nil) .{
                            .node = statement_return.expression,
                        } else .{
                            .token = statement_return.return_token,
                        },
                        .data = .{
                            .type_mismatch = .{
                                .lhs_type = procedure.general_data.return_type,
                                .rhs_type = expr_type,
                            },
                        },
                    });

                    return error.TypeMismatch;
                },
                else => |res| res,
            };
        },
        else => @panic(@tagName(statement_node.tag)),
    }
}

///Handles function overloading
pub fn resolveProcedure(
    self: *Sema,
    ast: Ast,
    name: Ast.NodeIndex,
    param_or_arg_list: Ast.NodeIndex,
) anyerror!*Procedure {
    const name_data: Ast.Node.Identifier = ast.dataFromNode(name, .expression_identifier);

    var type_buffer: [16]TypeIndex = undefined;
    var arg_node_buffer: [16]Ast.NodeIndex = undefined;

    const arg_type_list, const arg_node_list = try self.analyseArgList(
        ast,
        param_or_arg_list,
        .{
            .type_buffer = &type_buffer,
            .node_buffer = &arg_node_buffer,
        },
    );

    const procedure_overloads = self.procedure_overloads.getPtr(ast.tokenString(name_data.token)) orelse {
        try self.errors.append(self.gpa, .{
            .tag = .undeclared_identifier,
            .anchor = .{ .token = name_data.token },
        });

        return error.UndeclaredIdentifier;
    };

    if (arg_node_list.len < procedure_overloads.min_parameter_count or arg_type_list.len > procedure_overloads.max_parameter_count) {
        try self.errors.append(self.gpa, .{
            .tag = .argument_count_out_of_range,
            .anchor = .{ .node = param_or_arg_list },
            .data = .{
                .argument_count_out_of_range = .{
                    .expected_min_count = @intCast(procedure_overloads.min_parameter_count),
                    .expected_max_count = @intCast(procedure_overloads.max_parameter_count),
                    .actual_argument_count = @intCast(arg_type_list.len),
                },
            },
        });

        return error.ArgumentCountMismatch;
    }

    var matched_procedure: ?ProcedureIndex = null;

    if (procedure_overloads.overloads.entries.len == 1) {
        matched_procedure = procedure_overloads.overloads.values()[0];
    }

    if (matched_procedure == null) {
        matched_procedure = procedure_overloads.overloads.get(arg_type_list);
    }

    if (matched_procedure == null) {
        try self.errors.append(self.gpa, .{
            .tag = .no_matching_overload,
            .anchor = .{ .node = name },
        });

        return error.NoMatchingOverload;
    }

    const procedure_index = matched_procedure.?;

    //Resolve overload
    const procedure_definition = &self.procedures.items[@intFromEnum(procedure_index)];

    if (arg_type_list.len != procedure_definition.parameter_types.len) {
        try self.errors.append(self.gpa, .{
            .tag = .argument_count_mismatch,
            .anchor = .{ .node = param_or_arg_list },
            .data = .{
                .argument_count_mismatch = .{
                    .expected_argument_count = @intCast(procedure_definition.parameter_types.len),
                    .actual_argument_count = @intCast(arg_type_list.len),
                },
            },
        });

        return error.ArgumentCountMismatch;
    }

    for (arg_type_list, arg_node_list, 0..) |arg_type, arg_node, parameter_index| {
        const parameter_type_index = procedure_definition.parameter_types[parameter_index];

        const resultant_type = switch (coerceTypeAssign(parameter_type_index, arg_type)) {
            .null => {
                try self.errors.append(self.gpa, .{
                    .tag = .type_mismatch,
                    .anchor = .{ .node = arg_node },
                    .data = .{
                        .type_mismatch = .{
                            .lhs_type = parameter_type_index,
                            .rhs_type = arg_type,
                        },
                    },
                });

                return error.TypeMismatch;
            },
            else => |res| res,
        };
        _ = resultant_type; // autofix
    }

    //TODO: This is a fucking hack
    procedure_definition.general_data = procedure_overloads;

    return procedure_definition;
}

///Returns a list of positional argument types
pub fn analyseArgList(
    self: *Sema,
    ast: Ast,
    node: Ast.NodeIndex,
    state: struct {
        type_buffer: []TypeIndex,
        node_buffer: []Ast.NodeIndex,
        base_position: usize = 0,
    },
) !struct { []TypeIndex, []Ast.NodeIndex } {
    switch (node.tag) {
        .expression_binary_comma => {
            const binary_comma: Ast.Node.BinaryExpression = ast.dataFromNode(node, .expression_binary_comma);

            const new_position, _ = try self.analyseArgList(
                ast,
                binary_comma.left,
                .{
                    .type_buffer = state.type_buffer,
                    .node_buffer = state.node_buffer,
                },
            );

            return try self.analyseArgList(ast, binary_comma.right, .{
                .type_buffer = state.type_buffer,
                .node_buffer = state.node_buffer,
                .base_position = new_position.len,
            });
        },
        else => {
            const position = state.base_position;

            const expr_type = try self.resolveTypeFromExpression(ast, node);

            var expr_type_prim: TypeIndex.TypeIndexData = @bitCast(@intFromEnum(expr_type));

            //This canonicalizes primitive types like literal_uint -> uint
            if (expr_type.toArrayIndex() == null) {
                expr_type_prim.literal = 0;
            }

            state.type_buffer[position] = @enumFromInt(@as(u64, @bitCast(expr_type_prim)));
            state.node_buffer[position] = node;

            return .{ state.type_buffer[0 .. position + 1], state.node_buffer[0 .. position + 1] };
        },
    }
}

pub fn scopePush(self: *Sema) !void {
    try self.scope_stack.append(self.gpa, .{});
}

pub fn scopePop(self: *Sema) void {
    if (self.scope_stack.items.len == 0) {
        return;
    }

    const scope = &self.scope_stack.items[self.scope_stack.items.len - 1];

    scope.identifiers.deinit(self.gpa);

    self.scope_stack.items.len -= 1;
}

pub fn scopeDefine(
    self: *Sema,
    ast: Ast,
    identifier_token: Ast.TokenIndex,
    type_index: TypeIndex,
    qualifier: ValueQualifier,
    initial_value: ?u64,
) !void {
    const scope = &self.scope_stack.items[self.scope_stack.items.len - 1];

    const identifier_string = ast.tokenString(identifier_token);

    if (scope.identifiers.get(identifier_string)) |original_definition| {
        try self.errors.append(self.gpa, .{
            .tag = .identifier_redefined,
            .anchor = .{ .token = identifier_token },
            .data = .{
                .identifier_redefined = .{
                    .redefinition_identifier = identifier_token,
                    .definition_identifier = original_definition.token_index,
                },
            },
        });

        return error.IdentifierAlreadyDefined;
    }

    try scope.identifiers.put(self.gpa, identifier_string, .{
        .type_index = type_index,
        .token_index = identifier_token,
        .qualifier = qualifier,
        .initial_value = initial_value,
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

    try self.errors.append(self.gpa, .{
        .tag = .undeclared_identifier,
        .anchor = .{ .token = identifier_token },
    });

    return error.UndeclaredIdentifier;
}

pub fn resolveTypeFromTypeExpr(self: *Sema, ast: Ast, type_expr: Ast.NodeIndex) !TypeIndex {
    const type_expr_data: Ast.Node.TypeExpr = ast.dataFromNode(type_expr, .type_expr);

    return self.resolveTypeFromIdentifier(ast, type_expr_data.token);
}

pub fn resolveTypeFromIdentifier(self: *Sema, ast: Ast, type_expr_token: Ast.TokenIndex) !TypeIndex {
    const token_tag: Token.Tag = type_expr_token.tag;

    return switch (token_tag) {
        .keyword_uint => .uint,
        .keyword_int => .int,
        .keyword_float => .float,
        .keyword_double => .double,
        .keyword_bool => .bool,
        .keyword_void => .void,

        .keyword_vec2 => .vec2,
        .keyword_vec3 => .vec3,
        .keyword_vec4 => .vec4,

        .keyword_uvec2 => .uvec2,
        .keyword_uvec3 => .uvec3,
        .keyword_uvec4 => .uvec4,

        .keyword_ivec2 => .ivec2,
        .keyword_ivec3 => .ivec3,
        .keyword_ivec4 => .ivec4,

        .keyword_bvec2 => .bvec2,
        .keyword_bvec3 => .bvec3,
        .keyword_bvec4 => .bvec4,

        .keyword_dvec2 => .dvec2,
        .keyword_dvec3 => .dvec3,
        .keyword_dvec4 => .dvec4,

        .keyword_mat2, .keyword_mat2x2 => .mat2,
        .keyword_mat3, .keyword_mat3x3 => .mat3,
        .keyword_mat4, .keyword_mat4x4 => .mat4,
        .keyword_mat2x3 => .mat2x3,
        .keyword_mat3x4 => .mat3x4,
        .keyword_mat3x2 => .mat3x2,
        .keyword_mat4x3 => .mat4x3,
        .keyword_dmat2, .keyword_dmat2x2 => .dmat2,
        .keyword_dmat3, .keyword_dmat3x3 => .dmat3,
        .keyword_dmat4, .keyword_dmat4x4 => .dmat4,
        .keyword_dmat2x3 => .dmat2x3,
        .keyword_dmat3x4 => .dmat3x4,
        .keyword_dmat3x2 => .dmat3x2,
        .keyword_dmat4x3 => .dmat4x3,

        .identifier => {
            const type_index = self.type_map.get(ast.tokenString(type_expr_token)) orelse {
                try self.errors.append(self.gpa, .{
                    .tag = .undeclared_identifier,
                    .anchor = .{ .token = type_expr_token },
                });

                return error.UndeclaredIdentifier;
            };

            return type_index;
        },
        else => unreachable,
    };
}

pub fn resolveTypeFromIdentifierNoError(self: *Sema, ast: Ast, type_expr_token: Ast.TokenIndex) ?TypeIndex {
    const token_tag: Token.Tag = type_expr_token.tag;

    return switch (token_tag) {
        .keyword_uint => .uint,
        .keyword_int => .int,
        .keyword_float => .float,
        .keyword_double => .double,
        .keyword_bool => .bool,
        .keyword_void => .void,

        .keyword_vec2 => .vec2,
        .keyword_vec3 => .vec3,
        .keyword_vec4 => .vec4,

        .keyword_uvec2 => .uvec2,
        .keyword_uvec3 => .uvec3,
        .keyword_uvec4 => .uvec4,

        .keyword_ivec2 => .ivec2,
        .keyword_ivec3 => .ivec3,
        .keyword_ivec4 => .ivec4,

        .keyword_bvec2 => .bvec2,
        .keyword_bvec3 => .bvec3,
        .keyword_bvec4 => .bvec4,

        .keyword_dvec2 => .dvec2,
        .keyword_dvec3 => .dvec3,
        .keyword_dvec4 => .dvec4,

        .keyword_mat2, .keyword_mat2x2 => .mat2,
        .keyword_mat3, .keyword_mat3x3 => .mat3,
        .keyword_mat4, .keyword_mat4x4 => .mat4,
        .keyword_mat2x3 => .mat2x3,
        .keyword_mat3x4 => .mat3x4,
        .keyword_mat3x2 => .mat3x2,
        .keyword_mat4x3 => .mat4x3,
        .keyword_dmat2, .keyword_dmat2x2 => .dmat2,
        .keyword_dmat3, .keyword_dmat3x3 => .dmat3,
        .keyword_dmat4, .keyword_dmat4x4 => .dmat4,
        .keyword_dmat2x3 => .dmat2x3,
        .keyword_dmat3x4 => .dmat3x4,
        .keyword_dmat3x2 => .dmat3x2,
        .keyword_dmat4x3 => .dmat4x3,

        .identifier => {
            const type_index = self.type_map.get(ast.tokenString(type_expr_token)) orelse {
                return null;
            };

            return type_index;
        },
        else => @panic(@tagName(token_tag)),
    };
}

pub fn resolveTypeFromExpression(self: *Sema, ast: Ast, expression: Ast.NodeIndex) !TypeIndex {
    switch (expression.tag) {
        .expression_literal_boolean => return .literal_bool,
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
        .expression_binary_eql,
        .expression_binary_neql,
        .expression_binary_lt,
        .expression_binary_gt,
        .expression_binary_leql,
        .expression_binary_geql,
        => {
            const binary_expr: Ast.Node.BinaryExpression = ast.dataFromNode(expression, .expression_binary_add);

            const lhs_type = try self.resolveTypeFromExpression(ast, binary_expr.left);
            const rhs_type = try self.resolveTypeFromExpression(ast, binary_expr.right);

            const comparison_type = switch (coerceType(lhs_type, rhs_type)) {
                .null => {
                    try self.errors.append(self.gpa, .{
                        .tag = .type_mismatch,
                        .anchor = .{ .token = binary_expr.op_token },
                        .data = .{
                            .type_mismatch = .{
                                .lhs_type = lhs_type,
                                .rhs_type = rhs_type,
                            },
                        },
                    });

                    return error.TypeMismatch;
                },
                else => |res| res,
            };

            if (!comparison_type.isOperatorDefined(expression.tag)) {
                try self.errors.append(self.gpa, .{
                    .tag = .type_incompatibility,
                    .anchor = .{ .node = expression },
                    .data = .{
                        .type_incompatibility = .{
                            .lhs_type = lhs_type,
                            .rhs_type = rhs_type,
                        },
                    },
                });

                return error.TypeIncompatibilty;
            }

            return .bool;
        },
        .expression_binary_add,
        .expression_binary_sub,
        .expression_binary_mul,
        .expression_binary_div,
        .expression_binary_bitwise_xor,
        .expression_binary_bitwise_shift_left,
        .expression_binary_bitwise_shift_right,
        => {
            const binary_expr: Ast.Node.BinaryExpression = ast.dataFromNode(expression, .expression_binary_add);

            const lhs_type = try self.resolveTypeFromExpression(ast, binary_expr.left);
            const rhs_type = try self.resolveTypeFromExpression(ast, binary_expr.right);

            const coerced_type: TypeIndex = switch (expression.tag) {
                .expression_binary_mul, .expression_binary_div => coerceTypeMul(lhs_type, rhs_type),
                else => coerceType(lhs_type, rhs_type),
            };

            const resultant_type = switch (coerced_type) {
                .null => {
                    try self.errors.append(self.gpa, .{
                        .tag = .type_mismatch,
                        .anchor = .{ .node = expression },
                        .data = .{
                            .type_mismatch = .{
                                .lhs_type = lhs_type,
                                .rhs_type = rhs_type,
                            },
                        },
                    });

                    return error.TypeMismatch;
                },
                else => |res| res,
            };

            if (!resultant_type.isOperatorDefined(expression.tag)) {
                try self.errors.append(self.gpa, .{
                    .tag = .type_incompatibility,
                    .anchor = .{ .node = expression },
                    .data = .{
                        .type_incompatibility = .{
                            .lhs_type = lhs_type,
                            .rhs_type = rhs_type,
                        },
                    },
                });

                return error.TypeIncompatibilty;
            }

            return resultant_type;
        },
        .expression_binary_assign,
        .expression_binary_assign_add,
        .expression_binary_assign_sub,
        .expression_binary_assign_mul,
        .expression_binary_assign_div,
        .expression_binary_assign_bitwise_shift_left,
        .expression_binary_assign_bitwise_shift_right,
        => {
            const binary_expr: Ast.Node.BinaryExpression = ast.dataFromNode(expression, .expression_binary_add);

            const lhs_type = try self.resolveTypeFromExpression(ast, binary_expr.left);
            const rhs_type = try self.resolveTypeFromExpression(ast, binary_expr.right);

            const assignable = try self.isExpressionAssignable(ast, binary_expr.left);

            if (!assignable) {
                try self.errors.append(self.gpa, .{
                    .tag = .modified_const,
                    .anchor = .{ .node = binary_expr.left },
                });

                return error.ModifiedConstant;
            }

            const resultant_type = switch (coerceTypeAssign(lhs_type, rhs_type)) {
                .null => {
                    try self.errors.append(self.gpa, .{
                        .tag = .type_mismatch,
                        .anchor = .{ .token = binary_expr.op_token },
                        .data = .{
                            .type_mismatch = .{
                                .lhs_type = lhs_type,
                                .rhs_type = rhs_type,
                            },
                        },
                    });

                    return error.TypeMismatch;
                },
                else => |res| res,
            };

            if (!resultant_type.isOperatorDefined(expression.tag)) {
                try self.errors.append(self.gpa, .{
                    .tag = .type_incompatibility,
                    .anchor = .{ .node = expression },
                    .data = .{
                        .type_incompatibility = .{
                            .lhs_type = lhs_type,
                            .rhs_type = rhs_type,
                        },
                    },
                });

                return error.TypeIncompatibilty;
            }

            return resultant_type;
        },
        .expression_unary_minus => {
            const binary_expr: Ast.Node.BinaryExpression = ast.dataFromNode(expression, .expression_unary_minus);

            const rhs_type = try self.resolveTypeFromExpression(ast, binary_expr.right);

            if (!rhs_type.isOperatorDefined(expression.tag)) {
                try self.errors.append(self.gpa, .{
                    .tag = .type_incompatibility,
                    .anchor = .{ .node = expression },
                    .data = .{
                        .type_incompatibility = .{
                            .lhs_type = rhs_type,
                            .rhs_type = rhs_type,
                        },
                    },
                });

                return error.TypeIncompatibilty;
            }

            return rhs_type;
        },
        .expression_identifier => {
            const identifier: Ast.Node.Identifier = ast.dataFromNode(expression, .expression_identifier);

            const definition = try self.scopeResolve(ast, identifier.token);

            if (definition.initial_value != null) {
                var result_type = definition.type_index.toData();

                result_type.literal = 1;

                return @enumFromInt(@as(u64, @bitCast(result_type)));
            }

            return definition.type_index;
        },
        .expression_binary_proc_call => {
            const binary_expr: Ast.Node.BinaryExpression = ast.dataFromNode(expression, .expression_binary_proc_call);

            const identifier = binary_expr.left;

            const identifier_data: Ast.Node.Identifier = ast.dataFromNode(identifier, .expression_identifier);

            const type_for_constructor: TypeIndex = self.resolveTypeFromIdentifierNoError(ast, identifier_data.token) orelse .null;

            if (type_for_constructor != .null) {
                return type_for_constructor;
            }

            const arg_list = binary_expr.right;

            const procedure = try self.resolveProcedure(ast, identifier, arg_list);

            return procedure.general_data.return_type;
        },
        .expression_binary_field_access => {
            const binary_expr: Ast.Node.BinaryExpression = ast.dataFromNode(expression, .expression_binary_field_access);

            const lhs_type = try self.resolveTypeFromExpression(ast, binary_expr.left);
            const rhs_identifier: Ast.Node.Identifier = ast.dataFromNode(binary_expr.right, .expression_identifier);

            //TODO: support vectors
            if (lhs_type.toArrayIndex() == null) {
                try self.errors.append(self.gpa, .{
                    .anchor = .{ .node = expression },
                    .tag = .cannot_perform_field_access,
                    .data = .{
                        .cannot_perform_field_access = .{ .type_index = lhs_type },
                    },
                });

                return error.CannotPeformFieldAccess;
            }

            const type_data = self.types.items[lhs_type.toArrayIndex().?];

            const maybe_field = type_data.@"struct".fields.get(ast.tokenString(rhs_identifier.token));

            if (maybe_field) |field| {
                return field.type_index;
            }

            try self.errors.append(self.gpa, .{
                .anchor = .{
                    .token = rhs_identifier.token,
                },
                .tag = .no_field_in_struct,
                .data = .{
                    .no_field_in_struct = .{
                        .struct_type = lhs_type,
                    },
                },
            });

            return error.NoFieldInStruct;
        },
        .expression_binary_array_access => {
            const binary_expr: Ast.Node.BinaryExpression = ast.dataFromNode(expression, .expression_binary_array_access);

            const lhs_type = try self.resolveTypeFromExpression(ast, binary_expr.left);
            const rhs_type = try self.resolveTypeFromExpression(ast, binary_expr.right);

            if (!lhs_type.isArray()) {
                //TODO: throw error

                try self.errors.append(self.gpa, .{
                    .anchor = .{ .node = binary_expr.left },
                    .tag = .expression_not_indexable,
                });

                return error.ExpressionNotIndexable;
            }

            if (rhs_type.toData().literal == 1) {
                const array_index = try self.resolveConstantExpression(ast, binary_expr.right);

                if (array_index >= lhs_type.toData().array_length) {
                    try self.errors.append(self.gpa, .{
                        .anchor = .{ .node = binary_expr.right },
                        .tag = .array_access_out_of_bounds,
                        .data = .{ .array_index_out_of_bounds = .{
                            .array_length = lhs_type.toData().array_length,
                            .index = @intCast(array_index),
                        } },
                    });

                    return error.ArrayIndexOutOfBounds;
                }
            }

            return lhs_type.arrayElemType();
        },
        else => @panic(@tagName(expression.tag)),
    }
}

pub fn resolveConstantExpression(self: *Sema, ast: Ast, expression: Ast.NodeIndex) !u64 {
    switch (expression.tag) {
        .expression_binary_add,
        .expression_binary_sub,
        .expression_binary_mul,
        .expression_binary_div,
        => {
            const add_expr: Ast.Node.BinaryExpression = ast.dataFromNode(expression, .expression_binary_add);

            const lhs = try self.resolveConstantExpression(ast, add_expr.left);
            const rhs = try self.resolveConstantExpression(ast, add_expr.right);

            return switch (expression.tag) {
                .expression_binary_add => lhs + rhs,
                .expression_binary_sub => lhs - rhs,
                .expression_binary_mul => lhs * rhs,
                .expression_binary_div => lhs / rhs,
                else => unreachable,
            };
        },
        .expression_literal_number => {
            const literal_number_node: Ast.Node.ExpressionLiteralNumber = ast.dataFromNode(expression, .expression_literal_number);

            const value = try std.fmt.parseInt(u64, ast.tokenString(literal_number_node.token), 0);

            return value;
        },
        .expression_identifier => {
            const identifier_node: Ast.Node.Identifier = ast.dataFromNode(expression, .expression_identifier);

            const resolved_definition = try self.scopeResolve(ast, identifier_node.token);

            return resolved_definition.initial_value orelse return error.ConstantEvaluationFailed;
        },
        else => @panic("TODO: implement error message"),
    }
}

///Returns true if expression is an l-value
pub fn isExpressionAssignable(self: *Sema, ast: Ast, expression: Ast.NodeIndex) !bool {
    switch (expression.tag) {
        .expression_identifier => {
            const identifier_expression: Ast.Node.Identifier = ast.dataFromNode(expression, .expression_identifier);

            const identifier = try self.scopeResolve(ast, identifier_expression.token);

            return identifier.qualifier != .constant;
        },
        .expression_binary_field_access => {
            const field_access: Ast.Node.BinaryExpression = ast.dataFromNode(expression, .expression_binary_field_access);

            return self.isExpressionAssignable(ast, field_access.left);
        },
        .expression_binary_array_access => {
            const array_access: Ast.Node.BinaryExpression = ast.dataFromNode(expression, .expression_binary_array_access);

            return self.isExpressionAssignable(ast, array_access.left);
        },
        else => return false,
    }
}

pub fn coerceTypeAssign(lhs: TypeIndex, rhs: TypeIndex) TypeIndex {
    const result_type = coerceType(lhs, rhs);
    const lhs_primitive: TypeIndex.TypeIndexData = @bitCast(@intFromEnum(lhs));
    const type_primitive: TypeIndex.TypeIndexData = @bitCast(@intFromEnum(result_type));

    if (lhs_primitive.scalar_type == .integer and type_primitive.scalar_type != .integer) {
        return .null;
    }

    if (lhs_primitive.scalar_type == .float and type_primitive.scalar_type == .double) {
        return .null;
    }

    return result_type;
}

pub fn coerceTypeMul(lhs: TypeIndex, rhs: TypeIndex) TypeIndex {
    const simple_coerced_type = coerceType(lhs, rhs);

    if (simple_coerced_type != .null) {
        return simple_coerced_type;
    }

    const lhs_type_data: TypeIndex.TypeIndexData = @bitCast(@intFromEnum(lhs));
    const rhs_type_data: TypeIndex.TypeIndexData = @bitCast(@intFromEnum(rhs));

    const scalar_coercion = coerceType(lhs.componentScalar(), rhs.componentScalar());

    if (scalar_coercion == .null) {
        return .null;
    }

    var result_type: TypeIndex.TypeIndexData = @bitCast(@intFromEnum(scalar_coercion));

    if (lhs.isScalar() or rhs.isScalar()) {
        result_type.matrix_row_count = @max(lhs_type_data.matrix_row_count, rhs_type_data.matrix_row_count);
        result_type.matrix_column_count = @max(lhs_type_data.matrix_column_count, rhs_type_data.matrix_column_count);

        return @enumFromInt(@as(u64, @bitCast(result_type)));
    }

    return .null;
    // vecn = scalar * vecn
    // matnxk = scalar * matnxk
}

pub fn coerceTypeAddOrSub(lhs: TypeIndex, rhs: TypeIndex) TypeIndex {
    const result_type = coerceType(lhs, rhs);
    const lhs_primitive: TypeIndex.TypeIndexData = @bitCast(@intFromEnum(lhs));
    _ = lhs_primitive; // autofix
    const type_primitive: TypeIndex.TypeIndexData = @bitCast(@intFromEnum(result_type));
    _ = type_primitive; // autofix

    return result_type;
}

pub fn coerceType(lhs: TypeIndex, rhs: TypeIndex) TypeIndex {
    if (lhs == rhs) {
        return lhs;
    }

    if (lhs.toArrayIndex() != null or rhs.toArrayIndex() != null) {
        return .null;
    }

    const lhs_primitive: TypeIndex.TypeIndexData = @bitCast(@intFromEnum(lhs));
    const rhs_primitive: TypeIndex.TypeIndexData = @bitCast(@intFromEnum(rhs));

    if (lhs_primitive.array_length > 0 or rhs_primitive.array_length > 0) {
        return .null;
    }

    if (lhs_primitive.matrix_row_count != rhs_primitive.matrix_row_count or
        lhs_primitive.matrix_column_count != rhs_primitive.matrix_column_count)
    {
        return .null;
    }

    if (lhs_primitive.scalar_type == .bool or rhs_primitive.scalar_type == .bool) {
        if (lhs_primitive.scalar_type != rhs_primitive.scalar_type) {
            return .null;
        }
    }

    var result_type: TypeIndex.TypeIndexData = lhs_primitive;

    result_type.literal = lhs_primitive.literal & rhs_primitive.literal;

    if (lhs_primitive.null_or_void > 0 or rhs_primitive.null_or_void > 0) {
        if (lhs == .void or rhs == .void) {
            return .null;
        }
    }

    if (lhs_primitive.scalar_type == .integer and rhs_primitive.scalar_type == .integer) {
        result_type.sign = lhs_primitive.sign | rhs_primitive.sign;
    } else {
        const lhs_scalar_type_int: u2 = @intFromEnum(lhs_primitive.scalar_type);
        const rhs_scalar_type_int: u2 = @intFromEnum(rhs_primitive.scalar_type);

        result_type.sign = 0;
        result_type.scalar_type = @enumFromInt(lhs_scalar_type_int | rhs_scalar_type_int);
    }

    return @enumFromInt(@as(u64, @bitCast(result_type)));
}

pub fn printTypeName(self: Sema, ast: Ast, writer: *std.Io.Writer, type_index: TypeIndex) !void {
    if (type_index.isArray()) {
        try self.printTypeName(ast, writer, type_index.arrayElemType());

        try writer.print("[{}]", .{type_index.toData().array_length});

        return;
    }

    if (type_index.toArrayIndex() == null) {
        return try writer.writeAll(@tagName(type_index));
    }

    const type_data = &self.types.items[type_index.toArrayIndex().?];

    switch (type_data.*) {
        .@"struct" => |struct_data| {
            //TODO: handle line continuation
            return try writer.writeAll(ast.tokenString(struct_data.name));
        },
    }
}

pub const TypeIndex = enum(u64) {
    ///Used for tagging erroring types such that we don't contaminate future error messages with bad error checking
    null = 0,
    void,

    uint = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_row_count = 0,
        .matrix_column_count = 0,
        .literal = 0,
        .scalar_type = .integer,
    }),
    int = @bitCast(TypeIndexData{
        .sign = 1,
        .matrix_row_count = 0,
        .matrix_column_count = 0,
        .literal = 0,
        .scalar_type = .integer,
    }),
    float = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_row_count = 0,
        .matrix_column_count = 0,
        .literal = 0,
        .scalar_type = .float,
    }),
    double = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_row_count = 0,
        .matrix_column_count = 0,
        .literal = 0,
        .scalar_type = .double,
    }),
    bool = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_row_count = 0,
        .matrix_column_count = 0,
        .literal = 0,
        .scalar_type = .bool,
    }),

    uvec2 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_row_count = 1,
        .matrix_column_count = 0,
        .literal = 0,
        .scalar_type = .integer,
    }),
    uvec3 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_row_count = 2,
        .matrix_column_count = 0,
        .literal = 0,
        .scalar_type = .integer,
    }),
    uvec4 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_row_count = 3,
        .matrix_column_count = 0,
        .literal = 0,
        .scalar_type = .integer,
    }),

    ivec2 = @bitCast(TypeIndexData{
        .sign = 1,
        .matrix_row_count = 1,
        .matrix_column_count = 0,
        .literal = 0,
        .scalar_type = .integer,
    }),
    ivec3 = @bitCast(TypeIndexData{
        .sign = 1,
        .matrix_row_count = 2,
        .matrix_column_count = 0,
        .literal = 0,
        .scalar_type = .integer,
    }),
    ivec4 = @bitCast(TypeIndexData{
        .sign = 1,
        .matrix_row_count = 3,
        .matrix_column_count = 0,
        .literal = 0,
        .scalar_type = .integer,
    }),

    bvec2 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_row_count = 1,
        .matrix_column_count = 0,
        .literal = 0,
        .scalar_type = .bool,
    }),
    bvec3 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_row_count = 2,
        .matrix_column_count = 0,
        .literal = 0,
        .scalar_type = .bool,
    }),
    bvec4 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_row_count = 3,
        .matrix_column_count = 0,
        .literal = 0,
        .scalar_type = .bool,
    }),

    vec2 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_row_count = 1,
        .matrix_column_count = 0,
        .literal = 0,
        .scalar_type = .float,
    }),
    vec3 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_row_count = 2,
        .matrix_column_count = 0,
        .literal = 0,
        .scalar_type = .float,
    }),
    vec4 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_row_count = 3,
        .matrix_column_count = 0,
        .literal = 0,
        .scalar_type = .float,
    }),

    dvec2 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_row_count = 1,
        .matrix_column_count = 0,
        .literal = 0,
        .scalar_type = .double,
    }),
    dvec3 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_row_count = 2,
        .matrix_column_count = 0,
        .literal = 0,
        .scalar_type = .double,
    }),
    dvec4 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_row_count = 3,
        .matrix_column_count = 0,
        .literal = 0,
        .scalar_type = .double,
    }),

    mat2 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_row_count = 1,
        .matrix_column_count = 1,
        .literal = 0,
        .scalar_type = .float,
    }),
    mat3 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_row_count = 2,
        .matrix_column_count = 2,
        .literal = 0,
        .scalar_type = .float,
    }),
    mat4 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_row_count = 3,
        .matrix_column_count = 3,
        .literal = 0,
        .scalar_type = .float,
    }),

    mat2x3 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_column_count = 1,
        .matrix_row_count = 2,
        .literal = 0,
        .scalar_type = .float,
    }),
    mat2x4 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_column_count = 1,
        .matrix_row_count = 3,
        .literal = 0,
        .scalar_type = .float,
    }),

    mat3x2 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_column_count = 2,
        .matrix_row_count = 1,
        .literal = 0,
        .scalar_type = .float,
    }),
    mat3x4 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_column_count = 2,
        .matrix_row_count = 3,
        .literal = 0,
        .scalar_type = .float,
    }),
    mat4x2 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_column_count = 3,
        .matrix_row_count = 1,
        .literal = 0,
        .scalar_type = .float,
    }),
    mat4x3 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_column_count = 3,
        .matrix_row_count = 2,
        .literal = 0,
        .scalar_type = .float,
    }),

    dmat2 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_row_count = 1,
        .matrix_column_count = 1,
        .literal = 0,
        .scalar_type = .double,
    }),
    dmat3 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_row_count = 2,
        .matrix_column_count = 2,
        .literal = 0,
        .scalar_type = .double,
    }),
    dmat4 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_row_count = 3,
        .matrix_column_count = 3,
        .literal = 0,
        .scalar_type = .double,
    }),

    dmat2x3 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_column_count = 1,
        .matrix_row_count = 2,
        .literal = 0,
        .scalar_type = .double,
    }),
    dmat3x4 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_column_count = 2,
        .matrix_row_count = 3,
        .literal = 0,
        .scalar_type = .double,
    }),
    dmat2x4 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_column_count = 1,
        .matrix_row_count = 3,
        .literal = 0,
        .scalar_type = .double,
    }),
    dmat4x3 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_column_count = 3,
        .matrix_row_count = 2,
        .literal = 0,
        .scalar_type = .double,
    }),

    dmat3x2 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_column_count = 2,
        .matrix_row_count = 1,
        .literal = 0,
        .scalar_type = .double,
    }),
    dmat4x2 = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_column_count = 3,
        .matrix_row_count = 1,
        .literal = 0,
        .scalar_type = .double,
    }),

    literal_uint = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_row_count = 0,
        .matrix_column_count = 0,
        .literal = 1,
        .scalar_type = .integer,
    }),
    literal_int = @bitCast(TypeIndexData{
        .sign = 1,
        .matrix_row_count = 0,
        .matrix_column_count = 0,
        .literal = 1,
        .scalar_type = .integer,
    }),
    literal_float = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_row_count = 0,
        .matrix_column_count = 0,
        .literal = 1,
        .scalar_type = .float,
    }),
    literal_bool = @bitCast(TypeIndexData{
        .sign = 0,
        .matrix_row_count = 0,
        .matrix_column_count = 0,
        .literal = 1,
        .scalar_type = .bool,
    }),

    _,

    pub const array_index_begin: u64 = @as(u64, @bitCast(
        TypeIndexData{
            .literal = 1,
            .sign = 1,
            .scalar_type = @enumFromInt(3),
            .matrix_row_count = 3,
            .matrix_column_count = 3,
        },
    )) + 1;

    pub fn fromArrayIndex(array_index: usize) TypeIndex {
        const result: TypeIndexData = .{
            .array_length = 0,
            .literal = 0,
            .sign = 0,
            .scalar_type = @enumFromInt(0),
            .matrix_row_count = 0,
            .matrix_column_count = 0,
            .aggregate_index = @intCast(array_index + 1),
        };

        return @enumFromInt(@as(u64, @bitCast(result)));
    }

    pub fn toArrayIndex(type_index: TypeIndex) ?usize {
        const type_data: TypeIndexData = @bitCast(@intFromEnum(type_index));

        if (type_data.aggregate_index == 0) return null;

        return @as(usize, type_data.aggregate_index) - 1;
    }

    pub fn isArray(type_index: TypeIndex) bool {
        const type_data: TypeIndexData = @bitCast(@intFromEnum(type_index));

        return type_data.array_length > 0;
    }

    pub fn isScalar(type_index: TypeIndex) bool {
        const result_type_data: TypeIndexData = @bitCast(@intFromEnum(type_index));

        return result_type_data.matrix_row_count == 0 and result_type_data.matrix_column_count == 0;
    }

    pub fn arrayElemType(type_index: TypeIndex) TypeIndex {
        var result_type_data: TypeIndexData = @bitCast(@intFromEnum(type_index));

        result_type_data.array_length = 0;

        return @enumFromInt(@as(u64, @bitCast(result_type_data)));
    }

    pub fn componentScalar(type_index: TypeIndex) TypeIndex {
        var result_type_data: TypeIndexData = @bitCast(@intFromEnum(type_index));

        result_type_data.matrix_row_count = 0;
        result_type_data.matrix_column_count = 0;

        return @enumFromInt(@as(u64, @bitCast(result_type_data)));
    }

    ///Constructs an array type from a element type and a length
    pub fn array(element_type: TypeIndex, length: u32) TypeIndex {
        var result_type_data: TypeIndexData = @bitCast(@intFromEnum(element_type));
        result_type_data.array_length = length;

        return @enumFromInt(@as(u64, @bitCast(result_type_data)));
    }

    pub fn toData(self: TypeIndex) TypeIndexData {
        return @bitCast(@intFromEnum(self));
    }

    pub fn isOperatorDefined(self: TypeIndex, tag: Ast.Node.Tag) bool {
        switch (tag) {
            .expression_binary_add,
            .expression_binary_sub,
            .expression_binary_mul,
            .expression_binary_div,
            .expression_binary_assign_add,
            .expression_binary_assign_sub,
            .expression_binary_assign_mul,
            .expression_binary_assign_div,
            .expression_binary_lt,
            .expression_binary_gt,
            .expression_binary_leql,
            .expression_binary_geql,
            .expression_unary_minus,
            => {
                if (self.toArrayIndex() != null) {
                    return false;
                }

                const primitive_type: TypeIndexData = @bitCast(@intFromEnum(self));

                if (primitive_type.null_or_void != 0b11 or
                    primitive_type.scalar_type == .bool)
                {
                    return false;
                }

                return true;
            },
            .expression_binary_bitwise_xor,
            .expression_binary_bitwise_shift_left,
            .expression_binary_bitwise_shift_right,
            .expression_binary_assign_bitwise_shift_left,
            .expression_binary_assign_bitwise_shift_right,
            => {
                if (self.toArrayIndex() != null) {
                    return false;
                }

                const primitive_type: TypeIndexData = @bitCast(@intFromEnum(self));

                if (primitive_type.null_or_void != 0b11 or
                    primitive_type.scalar_type != .integer)
                {
                    return false;
                }

                return true;
            },
            .expression_binary_assign,
            .expression_binary_eql,
            .expression_binary_neql,
            => return switch (self) {
                .void,
                .null,
                => false,
                else => true,
            },
            else => return false,
        }
    }

    pub const TypeIndexData = packed struct(u64) {
        //If this is zero, the type is not an array
        array_length: u32 = 0,
        //Used to represent null and void
        null_or_void: u2 = 0b11,
        literal: u1,
        sign: u1,
        scalar_type: ScalarType,
        ///Used to reprsent vector lengths
        matrix_row_count: u2,
        matrix_column_count: u2,
        //Index into the type array - 1
        aggregate_index: u22 = 0,

        pub const ScalarType = enum(u2) {
            integer = 0b00,
            bool = 0b01,
            float = 0b10,
            double = 0b11,
        };
    };
};

const std = @import("std");
const Ast = @import("Ast.zig");
const spirv = @import("../spirv.zig");
const token_map = @import("token_map.zig");
const Token = @import("Tokenizer.zig").Token;
const Sema = @This();
