//! Implements the semantic analysis stage of the frontend

allocator: std.mem.Allocator,
scope_stack: std.ArrayList(struct {
    identifiers: token_map.Map(IdentifierDefinition) = .{},
}) = .{},
procedures: token_map.Map(Procedure) = .{},
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

pub const Procedure = struct {
    return_type: TypeIndex,
    parameters: []const Parameter,

    pub const Parameter = struct {
        type_index: TypeIndex,
        qualifier: ValueQualifier,
    };
};

pub const IdentifierDefinition = struct {
    token_index: Ast.TokenIndex,
    type_index: TypeIndex,
    qualifier: ValueQualifier,
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

            try self.errors.append(self.allocator, .{
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
            self.allocator,
            ast.tokenString(field_node_data.name),
            .{ .type_index = type_expr, .node = field_node },
        );
    }

    if (self.type_map.get(ast.tokenString(definition.name))) |type_index| {
        const type_data = self.types.items[type_index.toArrayIndex().?];

        try self.errors.append(self.allocator, .{
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

    try self.types.append(self.allocator, .{
        .@"struct" = .{
            .name = definition.name,
            .fields = fields,
        },
    });

    try self.type_map.put(
        self.allocator,
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

    const procedure_definition_get_result = try self.procedures.getOrPut(self.allocator, ast.tokenString(procedure.name));
    const procedure_definition = procedure_definition_get_result.value_ptr;

    if (!procedure_definition_get_result.found_existing) {
        procedure_definition.* = .{
            .parameters = &.{},
            .return_type = .void,
        };
    }

    procedure_definition.return_type = try self.resolveTypeFromTypeExpr(ast, procedure.return_type);

    if (procedure.param_list != Ast.NodeIndex.nil) {
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

            try self.scopeDefine(
                ast,
                param.name,
                param_type,
                param_definition.qualifier,
            );
        }
    }

    //TODO: handle forward declaration
    if (procedure.body == Ast.NodeIndex.nil) return;

    try self.analyseStatement(
        ast,
        procedure_definition.*,
        procedure.body,
        .{ .block_new_scope = false },
    );
}

pub fn analyseStatement(
    self: *Sema,
    ast: Ast,
    procedure: Procedure,
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

            const type_index = try self.resolveTypeFromExpression(ast, statement_node);

            try self.scopeDefine(
                ast,
                var_init.identifier,
                type_index,
                switch (var_init.qualifier) {
                    .keyword_in => .in,
                    .keyword_const => .constant,
                    else => unreachable,
                },
            );
        },
        .statement_if => {
            const statement_if: Ast.Node.StatementIf = ast.dataFromNode(statement_node, .statement_if);

            const condition_type = try self.resolveTypeFromExpression(ast, statement_if.condition_expression);

            _ = switch (coerceType(.bool, condition_type)) {
                .null => {
                    try self.errors.append(self.allocator, .{
                        .tag = .type_mismatch,
                        .anchor = .{ .token = statement_if.if_token },
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
        => {
            _ = try self.resolveTypeFromExpression(ast, statement_node);
        },
        .statement_return => {
            const statement_return: Ast.Node.StatementReturn = ast.dataFromNode(statement_node, .statement_return);

            const expr_type = if (statement_return.expression != Ast.NodeIndex.nil) try self.resolveTypeFromExpression(ast, statement_return.expression) else .void;

            _ = switch (coerceTypeAssign(procedure.return_type, expr_type)) {
                .null => {
                    try self.errors.append(self.allocator, .{
                        .tag = .type_mismatch,
                        .anchor = .{ .node = statement_return.expression },
                        .data = .{
                            .type_mismatch = .{
                                .lhs_type = procedure.return_type,
                                .rhs_type = expr_type,
                            },
                        },
                    });

                    return error.TypeMismatch;
                },
                else => |res| res,
            };
        },
        else => unreachable,
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

    //TODO: handle procedure overloading
    const procedure_definition = self.procedures.getPtr(ast.tokenString(name_data.token)) orelse {
        try self.errors.append(self.allocator, .{
            .tag = .undeclared_identifier,
            .anchor = .{ .token = name_data.token },
        });

        return error.UndeclaredIdentifier;
    };

    var type_buffer: [16]TypeIndex = undefined;

    const arg_type_list = try self.analyseArgList(ast, procedure_definition.*, param_or_arg_list, .{
        .op_token = name_data.token,
        .type_buffer = &type_buffer,
    });

    if (arg_type_list.len < procedure_definition.parameters.len) {
        try self.errors.append(self.allocator, .{
            .tag = .argument_count_mismatch,
            .anchor = .{ .token = name_data.token },
            .data = .{
                .argument_count_mismatch = .{
                    .expected_argument_count = @intCast(procedure_definition.parameters.len),
                    .actual_argument_count = @intCast(arg_type_list.len),
                },
            },
        });

        return error.ArgumentCountMismatch;
    }

    return procedure_definition;
}

///Returns a list of positional argument types
pub fn analyseArgList(
    self: *Sema,
    ast: Ast,
    procedure: Procedure,
    node: Ast.NodeIndex,
    state: struct {
        type_buffer: []TypeIndex,
        base_position: usize = 0,
        op_token: Ast.TokenIndex,
    },
) ![]TypeIndex {
    switch (node.tag) {
        .expression_binary_comma => {
            const binary_comma: Ast.Node.BinaryExpression = ast.dataFromNode(node, .expression_binary_comma);

            const new_position = try self.analyseArgList(
                ast,
                procedure,
                binary_comma.left,
                .{ .type_buffer = state.type_buffer, .op_token = binary_comma.op_token },
            );

            return try self.analyseArgList(ast, procedure, binary_comma.right, .{
                .type_buffer = state.type_buffer,
                .base_position = new_position.len,
                .op_token = binary_comma.op_token,
            });
        },
        else => {
            const position = state.base_position;

            if (position >= procedure.parameters.len) {
                try self.errors.append(self.allocator, .{
                    .tag = .argument_count_mismatch,
                    .anchor = .{ .token = state.op_token },
                    .data = .{
                        .argument_count_mismatch = .{
                            .expected_argument_count = @intCast(procedure.parameters.len),
                            .actual_argument_count = @intCast(position + 1),
                        },
                    },
                });

                return error.ArgumentCountMismatch;
            }

            const parameter = &procedure.parameters[position];
            const expr_type = try self.resolveTypeFromExpression(ast, node);

            const resultant_type = switch (coerceTypeAssign(parameter.type_index, expr_type)) {
                .null => {
                    try self.errors.append(self.allocator, .{
                        .tag = .type_mismatch,
                        .anchor = .{ .token = state.op_token },
                        .data = .{
                            .type_mismatch = .{
                                .lhs_type = parameter.type_index,
                                .rhs_type = expr_type,
                            },
                        },
                    });

                    return error.TypeMismatch;
                },
                else => |res| res,
            };

            state.type_buffer[position] = resultant_type;

            return state.type_buffer[0 .. position + 1];
        },
    }
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

    self.scope_stack.items.len -= 1;
}

pub fn scopeDefine(
    self: *Sema,
    ast: Ast,
    identifier_token: Ast.TokenIndex,
    type_index: TypeIndex,
    qualifier: ValueQualifier,
) !void {
    const scope = &self.scope_stack.items[self.scope_stack.items.len - 1];

    const identifier_string = ast.tokenString(identifier_token);

    if (scope.identifiers.get(identifier_string)) |original_definition| {
        try self.errors.append(self.allocator, .{
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

    try scope.identifiers.put(self.allocator, identifier_string, .{
        .type_index = type_index,
        .token_index = identifier_token,
        .qualifier = qualifier,
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
        .anchor = .{ .token = identifier_token },
    });

    return error.UndeclaredIdentifier;
}

pub fn resolveTypeFromTypeExpr(self: *Sema, ast: Ast, type_expr: Ast.NodeIndex) !TypeIndex {
    const type_expr_data: Ast.Node.TypeExpr = ast.dataFromNode(type_expr, .type_expr);

    return self.resolveTypeFromIdentifier(ast, type_expr_data.token);
}

pub fn resolveTypeFromIdentifier(self: *Sema, ast: Ast, type_expr_token: Ast.TokenIndex) !TypeIndex {
    const token_tag: Token.Tag = ast.tokens.items(.tag)[@intFromEnum(type_expr_token)];

    return switch (token_tag) {
        .keyword_uint => .uint,
        .keyword_int => .int,
        .keyword_float => .float,
        .keyword_bool => .bool,
        .keyword_void => .void,
        .identifier => {
            const type_index = self.type_map.get(ast.tokenString(type_expr_token)) orelse {
                try self.errors.append(self.allocator, .{
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
    const token_tag: Token.Tag = ast.tokens.items(.tag)[@intFromEnum(type_expr_token)];

    return switch (token_tag) {
        .keyword_uint => .uint,
        .keyword_int => .int,
        .keyword_float => .float,
        .keyword_bool => .bool,
        .keyword_void => .void,
        .identifier => {
            const type_index = self.type_map.get(ast.tokenString(type_expr_token)) orelse {
                return null;
            };

            return type_index;
        },
        else => unreachable,
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
        .statement_var_init => {
            const var_init: Ast.Node.StatementVarInit = ast.dataFromNode(expression, .statement_var_init);

            const type_index = try self.resolveTypeFromTypeExpr(ast, var_init.type_expr);
            const expression_type_index = try self.resolveTypeFromExpression(ast, var_init.expression);

            const resultant_type = switch (coerceTypeAssign(type_index, expression_type_index)) {
                .null => {
                    try self.errors.append(self.allocator, .{
                        .tag = .type_mismatch,
                        .anchor = .{ .token = var_init.identifier },
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

            return resultant_type;
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
                    try self.errors.append(self.allocator, .{
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
                try self.errors.append(self.allocator, .{
                    .tag = .type_incompatibility,
                    .anchor = .{ .token = binary_expr.op_token },
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
        => {
            const binary_expr: Ast.Node.BinaryExpression = ast.dataFromNode(expression, .expression_binary_add);

            const lhs_type = try self.resolveTypeFromExpression(ast, binary_expr.left);
            const rhs_type = try self.resolveTypeFromExpression(ast, binary_expr.right);

            const resultant_type = switch (coerceType(lhs_type, rhs_type)) {
                .null => {
                    try self.errors.append(self.allocator, .{
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
                try self.errors.append(self.allocator, .{
                    .tag = .type_incompatibility,
                    .anchor = .{ .token = binary_expr.op_token },
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
        => {
            const binary_expr: Ast.Node.BinaryExpression = ast.dataFromNode(expression, .expression_binary_add);

            const assignable = try self.isExpressionAssignable(ast, binary_expr.left);

            if (!assignable) {
                try self.errors.append(self.allocator, .{
                    .tag = .modified_const,
                    .anchor = .{ .token = binary_expr.op_token },
                });

                return error.ModifiedConstant;
            }

            const lhs_type = try self.resolveTypeFromExpression(ast, binary_expr.left);
            const rhs_type = try self.resolveTypeFromExpression(ast, binary_expr.right);

            const resultant_type = switch (coerceTypeAssign(lhs_type, rhs_type)) {
                .null => {
                    try self.errors.append(self.allocator, .{
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
                try self.errors.append(self.allocator, .{
                    .tag = .type_incompatibility,
                    .anchor = .{ .token = binary_expr.op_token },
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
        .expression_identifier => {
            const identifier: Ast.Node.Identifier = ast.dataFromNode(expression, .expression_identifier);

            const definition = try self.scopeResolve(ast, identifier.token);

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

            return procedure.return_type;
        },
        else => unreachable,
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
        else => return false,
    }
}

pub fn coerceTypeAssign(lhs: TypeIndex, rhs: TypeIndex) TypeIndex {
    const result_type = coerceType(lhs, rhs);
    const lhs_primitive: TypeIndex.PrimitiveNumericType = @bitCast(@intFromEnum(lhs));
    const type_primitive: TypeIndex.PrimitiveNumericType = @bitCast(@intFromEnum(result_type));

    if (lhs_primitive.scalar_type == .integer and type_primitive.scalar_type != .integer) {
        return .null;
    }

    return result_type;
}

pub fn coerceType(lhs: TypeIndex, rhs: TypeIndex) TypeIndex {
    if (lhs == rhs) {
        return lhs;
    }

    if (lhs.toArrayIndex() != null or rhs.toArrayIndex() != null) {
        return .null;
    }

    const lhs_primitive: TypeIndex.PrimitiveNumericType = @bitCast(@intFromEnum(lhs));
    const rhs_primitive: TypeIndex.PrimitiveNumericType = @bitCast(@intFromEnum(rhs));

    if (lhs_primitive.linear_algebra_type != rhs_primitive.linear_algebra_type or
        lhs_primitive.component_count != rhs_primitive.component_count)
    {
        return .null;
    }

    if (lhs_primitive.scalar_type == .bool or rhs_primitive.scalar_type == .bool) {
        if (lhs_primitive.scalar_type != rhs_primitive.scalar_type) {
            return .null;
        }
    }

    var result_type: TypeIndex.PrimitiveNumericType = lhs_primitive;

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

    return @enumFromInt(@as(u32, @bitCast(result_type)));
}

pub fn typeName(self: Sema, ast: Ast, type_index: TypeIndex) []const u8 {
    if (type_index.toArrayIndex() == null) {
        return @tagName(type_index);
    }

    const type_data = &self.types.items[type_index.toArrayIndex().?];

    switch (type_data.*) {
        .@"struct" => |struct_data| {
            return ast.tokenString(struct_data.name);
        },
    }
}

pub const TypeIndex = enum(u32) {
    ///Used for tagging erroring types such that we don't contaminate future error messages with bad error checking
    null = 0,
    void,

    uint = @bitCast(PrimitiveNumericType{
        .sign = 0,
        .component_count = 0,
        .linear_algebra_type = .vector,
        .literal = 0,
        .scalar_type = .integer,
    }),
    int = @bitCast(PrimitiveNumericType{
        .sign = 1,
        .component_count = 0,
        .linear_algebra_type = .vector,
        .literal = 0,
        .scalar_type = .integer,
    }),
    float = @bitCast(PrimitiveNumericType{
        .sign = 0,
        .component_count = 0,
        .linear_algebra_type = .vector,
        .literal = 0,
        .scalar_type = .float,
    }),
    bool = @bitCast(PrimitiveNumericType{
        .sign = 0,
        .component_count = 0,
        .linear_algebra_type = .vector,
        .literal = 0,
        .scalar_type = .bool,
    }),

    literal_uint = @bitCast(PrimitiveNumericType{
        .sign = 0,
        .component_count = 0,
        .linear_algebra_type = .vector,
        .literal = 1,
        .scalar_type = .integer,
    }),
    literal_int = @bitCast(PrimitiveNumericType{
        .sign = 1,
        .component_count = 0,
        .linear_algebra_type = .vector,
        .literal = 1,
        .scalar_type = .integer,
    }),
    literal_float = @bitCast(PrimitiveNumericType{
        .sign = 0,
        .component_count = 0,
        .linear_algebra_type = .vector,
        .literal = 1,
        .scalar_type = .float,
    }),
    literal_bool = @bitCast(PrimitiveNumericType{
        .sign = 0,
        .component_count = 0,
        .linear_algebra_type = .vector,
        .literal = 1,
        .scalar_type = .bool,
    }),

    _,

    pub const array_index_begin: u32 = @as(u32, @bitCast(
        PrimitiveNumericType{
            .literal = 1,
            .sign = 1,
            .scalar_type = @enumFromInt(3),
            .linear_algebra_type = @enumFromInt(1),
            .component_count = 1,
        },
    )) + 1;

    pub fn fromArrayIndex(array_index: usize) TypeIndex {
        const integer = array_index_begin + array_index;

        return @enumFromInt(integer);
    }

    pub fn toArrayIndex(type_index: TypeIndex) ?usize {
        const integer = @intFromEnum(type_index);

        if (integer < array_index_begin) {
            return null;
        }

        return integer - array_index_begin;
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
            => {
                if (self.toArrayIndex() != null) {
                    return false;
                }

                const primitive_type: PrimitiveNumericType = @bitCast(@intFromEnum(self));

                if (primitive_type.null_or_void != 0b11 or
                    primitive_type.scalar_type == .bool)
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

    pub const PrimitiveNumericType = packed struct(u32) {
        //Used to represent null and void
        null_or_void: u2 = 0b11,
        literal: u1,
        sign: u1,
        scalar_type: ScalarType,
        linear_algebra_type: LinearAlgebraType,
        ///For vectors and matricies
        component_count: u2,
        padding1: u23 = 0,

        pub const ScalarType = enum(u2) {
            integer = 0b00,
            bool = 0b01,
            float = 0b10,
            double = 0b11,
        };

        pub const LinearAlgebraType = enum(u1) {
            vector,
            matrix,
        };
    };
};

const std = @import("std");
const Ast = @import("Ast.zig");
const spirv = @import("../spirv.zig");
const token_map = @import("token_map.zig");
const Token = @import("Tokenizer.zig").Token;
const Sema = @This();
