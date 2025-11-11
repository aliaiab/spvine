//! Implements the syntactic analysis stage of the frontend

gpa: std.mem.Allocator,
source: []const u8,
tokenizer: Tokenizer,
///Stores a centered slice of the token stream
///The centre (token_window[1]) is the current token
token_window: TokenWindow = [1]Ast.TokenIndex{.invalid} ** 3,
defines: DefineMap = .{},
///This is the generation of #define we're currently on
define_generation: u32 = 0,
directive_if_level: u32 = 0,
directive_if_condition: bool = false,
tokenizer_stack: std.ArrayList(struct {
    //This may be able to be inferred from the token window (bar the retained state within Tokenizer)
    tokenizer: Tokenizer,
    define_generation: u32,
}) = .{},
errors: std.ArrayList(Ast.Error),
node_heap: Ast.NodeHeap = .{},
root_decls: []Ast.NodeIndex,

pub fn init(
    allocator: std.mem.Allocator,
    source: []const u8,
) Parser {
    return .{
        .gpa = allocator,
        .source = source,
        .tokenizer = Tokenizer.init(source),
        .errors = .{},
        .root_decls = &.{},
    };
}

pub fn deinit(self: *Parser) void {
    defer self.* = undefined;
    defer self.errors.deinit(self.gpa);
    // defer self.node_heap.deinit(self.allocator);
}

///Root parse node
pub fn parse(self: *Parser) !void {
    var root_nodes: std.ArrayList(Ast.NodeIndex) = .{};
    defer root_nodes.deinit(self.gpa);

    //Init the token window
    self.token_window[0] = .invalid;
    self.token_window[1] = try self.advanceTokenizer();
    self.token_window[2] = try self.advanceTokenizer();

    while (self.peekToken().tag != .end_of_file) {
        switch (self.peekToken().tag) {
            .keyword_struct => {
                const struct_def = try self.parseStruct();

                try root_nodes.append(self.gpa, struct_def);
            },
            .keyword_float,
            .keyword_double,
            .keyword_uint,
            .keyword_int,
            .keyword_bool,
            .keyword_vec2,
            .keyword_vec3,
            .keyword_vec4,
            .keyword_ivec2,
            .keyword_ivec3,
            .keyword_ivec4,
            .keyword_uvec2,
            .keyword_uvec3,
            .keyword_uvec4,
            .keyword_bvec2,
            .keyword_bvec3,
            .keyword_bvec4,
            .keyword_dvec2,
            .keyword_dvec3,
            .keyword_dvec4,
            .keyword_mat2x2,
            .keyword_mat3x3,
            .keyword_mat4x4,
            .keyword_mat2x3,
            .keyword_mat3x4,
            .keyword_mat3x2,
            .keyword_mat4x3,
            .keyword_dmat2,
            .keyword_dmat3,
            .keyword_dmat4,
            .keyword_dmat2x2,
            .keyword_dmat3x3,
            .keyword_dmat4x4,
            .keyword_dmat2x3,
            .keyword_dmat3x4,
            .keyword_dmat3x2,
            .keyword_dmat4x3,
            .keyword_void,
            .identifier,
            => {
                const proc = try self.parseProcedure();

                try root_nodes.append(self.gpa, proc);
            },
            else => {
                _ = try self.nextToken();
            },
        }
    }

    self.root_decls = try root_nodes.toOwnedSlice(self.gpa);
}

pub fn parseStruct(self: *Parser) !Ast.NodeIndex {
    var node_index = try self.reserveNode(.struct_definition);
    errdefer self.unreserveNode(node_index);

    _ = try self.expectToken(.keyword_struct);

    const struct_name_identifier = try self.expectToken(.identifier);

    _ = try self.expectToken(.left_brace);

    var field_nodes: std.ArrayList(Ast.NodeIndex) = .{};
    defer field_nodes.deinit(self.gpa);

    while (self.peekTokenTag().? != .right_brace) {
        const field_type = try self.parseTypeExpr();

        const field_name = try self.expectToken(.identifier);

        var struct_field_node = try self.reserveNode(.struct_field);

        try self.nodeSetData(&struct_field_node, .struct_field, .{
            .name = field_name,
            .type_expr = field_type,
        });

        try field_nodes.append(self.gpa, struct_field_node);

        _ = try self.expectToken(.semicolon);
    }

    try self.nodeSetData(&node_index, .struct_definition, .{
        .name = struct_name_identifier,
        .fields = try self.node_heap.allocateDupe(self.gpa, Ast.NodeIndex, field_nodes.items),
    });

    _ = try self.expectToken(.right_brace);
    _ = try self.expectToken(.semicolon);

    return node_index;
}

pub fn parseProcedure(self: *Parser) !Ast.NodeIndex {
    var node_index = try self.reserveNode(.procedure);
    errdefer self.unreserveNode(node_index);

    const type_expr = try self.parseTypeExpr();

    const identifier = try self.expectToken(.identifier);

    _ = try self.expectToken(.left_paren);

    const param_list = try self.parseParamList();

    _ = try self.expectToken(.right_paren);

    var body = Ast.NodeIndex.nil;

    if (self.peekTokenTag() == .left_brace) {
        body = try self.parseStatement();
    } else {
        _ = try self.expectToken(.semicolon);
    }

    try self.nodeSetData(&node_index, .procedure, .{
        .return_type = type_expr,
        .name = identifier,
        .param_list = param_list,
        .body = body,
    });

    return node_index;
}

pub fn parseParamList(self: *Parser) !Ast.NodeIndex {
    var node = try self.reserveNode(.param_list);
    errdefer self.unreserveNode(node);

    var param_nodes: std.ArrayList(Ast.NodeIndex) = .{};
    defer param_nodes.deinit(self.gpa);

    while (self.peekTokenTag().? != .right_paren) {
        const param = try self.parseParam();

        try param_nodes.append(self.gpa, param);

        if (self.lookAheadTokenTag(1) != .right_paren) {
            _ = try self.eatToken(.comma);
        }
    }

    try self.nodeSetData(&node, .param_list, .{
        .params = try self.node_heap.allocateDupe(self.gpa, Ast.NodeIndex, param_nodes.items),
    });

    return node;
}

pub fn parseParam(self: *Parser) !Ast.NodeIndex {
    var node = try self.reserveNode(.param_expr);
    errdefer self.unreserveNode(node);

    const qualifier: Token.Tag = try self.eatVariableQualifier();

    const type_expr = try self.parseTypeExpr();
    const param_identifier = try self.expectToken(.identifier);

    try self.nodeSetData(&node, .param_expr, .{
        .type_expr = type_expr,
        .name = param_identifier,
        .qualifier = qualifier,
    });

    return node;
}

pub fn eatVariableQualifier(self: *Parser) !Token.Tag {
    var qualifier: Token.Tag = .keyword_in;

    const first_in_qualifier = try self.eatToken(.keyword_in);

    switch (self.peekTokenTag().?) {
        .keyword_inout,
        .keyword_out,
        .keyword_in,
        .keyword_const,
        => |tag| {
            qualifier = tag;

            _ = try self.nextToken();
        },
        else => {},
    }

    if (first_in_qualifier == null) {
        _ = try self.eatToken(.keyword_in);
    }

    return qualifier;
}

pub fn parseStatement(self: *Parser) !Ast.NodeIndex {
    switch (self.peekTokenTag().?) {
        .left_brace => {
            var node = try self.reserveNode(.statement_block);
            errdefer self.unreserveNode(node);

            _ = try self.nextToken();

            var statements: std.ArrayList(Ast.NodeIndex) = .{};
            defer statements.deinit(self.gpa);

            while (self.peekTokenTag().? != .right_brace) {
                const statement = try self.parseStatement();

                if (statement == Ast.NodeIndex.nil) continue;

                try statements.append(self.gpa, statement);
            }

            _ = try self.expectToken(.right_brace);

            try self.nodeSetData(&node, .statement_block, .{
                .statements = try self.node_heap.allocateDupe(self.gpa, Ast.NodeIndex, statements.items),
            });

            return node;
        },
        .left_paren => {
            return self.parseExpression(.{});
        },
        .keyword_const,
        .keyword_float,
        .keyword_double,
        .keyword_uint,
        .keyword_int,
        .keyword_bool,
        .keyword_vec2,
        .keyword_vec3,
        .keyword_vec4,
        .keyword_ivec2,
        .keyword_ivec3,
        .keyword_ivec4,
        .keyword_uvec2,
        .keyword_uvec3,
        .keyword_uvec4,
        .keyword_bvec2,
        .keyword_bvec3,
        .keyword_bvec4,
        .keyword_dvec2,
        .keyword_dvec3,
        .keyword_dvec4,
        .keyword_mat2x2,
        .keyword_mat3x3,
        .keyword_mat4x4,
        .keyword_mat2x3,
        .keyword_mat3x4,
        .keyword_mat3x2,
        .keyword_mat4x3,
        .keyword_dmat2,
        .keyword_dmat3,
        .keyword_dmat4,
        .keyword_dmat2x2,
        .keyword_dmat3x3,
        .keyword_dmat4x4,
        .keyword_dmat2x3,
        .keyword_dmat3x4,
        .keyword_dmat3x2,
        .keyword_dmat4x3,
        .keyword_void,
        .identifier,
        => |token_tag| {
            if (token_tag == .identifier) {
                const next_token = self.lookAheadTokenTag(1);

                if (next_token != .end_of_file) {
                    if (next_token != .identifier) {
                        const expr = self.parseExpression(.{});

                        _ = try self.eatToken(.semicolon);

                        return expr;
                    }
                }
            }

            var node = try self.reserveNode(.statement_var_init);
            errdefer self.unreserveNode(node);

            const qualifier = try self.eatVariableQualifier();

            const type_expr = try self.parseTypeExpr();

            const variable_name = try self.expectToken(.identifier);

            const array_left_bracket = try self.eatToken(.left_bracket);

            var array_length_specifier: Ast.NodeIndex = .nil;

            if (array_left_bracket != null) {
                //TODO: I don't know if parsing this an expression is too lenient.
                //I think it's fine to just let semantic analysis check if this is a constant
                const array_length_expr = try self.parseExpression(.{});

                array_length_specifier = array_length_expr;

                _ = try self.expectToken(.right_bracket);
            }

            if (try self.eatToken(.equals) != null) {
                const expression = try self.parseExpression(.{});

                try self.nodeSetData(&node, .statement_var_init, .{
                    .type_expr = type_expr,
                    .array_length_specifier = array_length_specifier,
                    .identifier = variable_name,
                    .expression = expression,
                    .qualifier = qualifier,
                });
            } else {
                try self.nodeSetData(&node, .statement_var_init, .{
                    .type_expr = type_expr,
                    .array_length_specifier = array_length_specifier,
                    .identifier = variable_name,
                    .qualifier = qualifier,
                    .expression = Ast.NodeIndex.nil,
                });
            }

            return node;
        },
        .literal_number,
        => {
            const expr = try self.parseExpression(.{});

            _ = try self.eatToken(.semicolon);

            return expr;
        },
        .literal_string,
        => {
            //TODO: what to do with string literals
            _ = try self.nextToken();

            @panic("unimplemented: string literals are not supported yet");
        },
        .keyword_if => {
            const if_token = try self.expectToken(.keyword_if);

            var node = try self.reserveNode(.statement_if);
            errdefer self.unreserveNode(node);

            _ = try self.expectToken(.left_paren);

            const cond_expr = try self.parseExpression(.{});

            _ = try self.expectToken(.right_paren);

            const taken_statment = try self.parseStatement();

            var not_taken_statment = Ast.NodeIndex.nil;

            const else_keyword = try self.eatToken(.keyword_else);

            if (else_keyword) |_| {
                not_taken_statment = try self.parseStatement();
            }

            try self.nodeSetData(&node, .statement_if, .{
                .if_token = if_token,
                .condition_expression = cond_expr,
                .taken_statement = taken_statment,
                .not_taken_statement = not_taken_statment,
            });

            return node;
        },
        .keyword_return => {
            const keyword_return = try self.nextToken();

            var node = try self.reserveNode(.statement_return);
            errdefer self.unreserveNode(node);

            var expression: Ast.NodeIndex = Ast.NodeIndex.nil;

            if (self.peekTokenTag().? != .semicolon) {
                expression = try self.parseExpression(.{});
            }

            _ = try self.expectToken(.semicolon);

            try self.nodeSetData(&node, .statement_return, .{
                .expression = expression,
                .return_token = keyword_return,
            });

            return node;
        },
        .semicolon => {
            _ = try self.nextToken();
            return Ast.NodeIndex.nil;
        },
        else => return self.unexpectedToken(),
    }

    return Ast.NodeIndex.nil;
}

pub fn parseExpression(
    self: *Parser,
    context: struct {
        min_precedence: i32 = std.math.minInt(i32),
        left: Ast.NodeIndex = Ast.NodeIndex.nil,
    },
) anyerror!Ast.NodeIndex {
    var lhs = context.left;

    while (true) {
        var node = Ast.NodeIndex.nil;
        errdefer if (node != Ast.NodeIndex.nil) self.unreserveNode(node);

        const binary = struct {
            pub inline fn getPrecedence(comptime node_tag: Ast.Node.Tag) i32 {
                return switch (node_tag) {
                    .expression_binary_field_access,
                    .expression_binary_array_access,
                    => 10,
                    .expression_unary_minus,
                    => 9,
                    .expression_binary_mul,
                    .expression_binary_div,
                    => 8,
                    .expression_binary_add,
                    .expression_binary_sub,
                    => 7,
                    .expression_binary_bitwise_shift_left,
                    .expression_binary_bitwise_shift_right,
                    => 6,
                    .expression_binary_lt,
                    .expression_binary_gt,
                    .expression_binary_leql,
                    .expression_binary_geql,
                    => 5,
                    .expression_binary_eql,
                    .expression_binary_neql,
                    => 4,
                    .expression_binary_bitwise_xor => 3,
                    .expression_binary_assign,
                    .expression_binary_assign_add,
                    .expression_binary_assign_sub,
                    .expression_binary_assign_mul,
                    .expression_binary_assign_div,
                    => 2,
                    .expression_binary_comma => 1,
                    else => 0,
                };
            }

            pub inline fn getNodeType(comptime token_tag: Token.Tag) ?Ast.Node.Tag {
                return switch (token_tag) {
                    .equals => .expression_binary_assign,
                    .plus_equals => .expression_binary_assign_add,
                    .minus_equals => .expression_binary_assign_sub,
                    .asterisk_equals => .expression_binary_assign_mul,
                    .forward_slash_equals => .expression_binary_assign_div,
                    .plus => .expression_binary_add,
                    .minus => .expression_binary_sub,
                    .asterisk => .expression_binary_mul,
                    .forward_slash => .expression_binary_div,
                    .equals_equals => .expression_binary_eql,
                    .bang_equals => .expression_binary_neql,
                    .left_angled_bracket => .expression_binary_lt,
                    .right_angled_bracket => .expression_binary_gt,
                    .less_than_equals => .expression_binary_leql,
                    .greater_than_equals => .expression_binary_geql,
                    .comma => .expression_binary_comma,
                    .unary_minus => .expression_unary_minus,
                    .period => .expression_binary_field_access,
                    .left_bracket => .expression_binary_array_access,
                    .caret => .expression_binary_bitwise_xor,
                    .double_left_angled_bracket => .expression_binary_bitwise_shift_left,
                    .double_right_angled_bracket => .expression_binary_bitwise_shift_right,
                    .double_left_angled_bracket_equals => .expression_binary_assign_bitwise_shift_left,
                    .double_right_angled_bracket_equals => .expression_binary_assign_bitwise_shift_right,
                    else => null,
                };
            }

            pub fn isBinaryExpression(token_tag: Token.Tag) bool {
                return switch (token_tag) {
                    inline else => |tag_comptime| {
                        return getNodeType(tag_comptime) != null;
                    },
                };
            }
        };

        switch (self.peekTokenTag().?) {
            .literal_number,
            => {
                if (lhs != Ast.NodeIndex.nil and !binary.isBinaryExpression(self.previousTokenTag())) {
                    return self.unexpectedToken();
                }

                node = try self.reserveNode(.expression_literal_number);

                const literal = try self.nextToken();

                try self.nodeSetData(&node, .expression_literal_number, .{
                    .token = literal,
                });
            },
            .keyword_true,
            .keyword_false,
            => {
                if (lhs != Ast.NodeIndex.nil and !binary.isBinaryExpression(self.previousTokenTag())) {
                    return self.unexpectedToken();
                }

                node = try self.reserveNode(.expression_literal_boolean);

                const literal = try self.nextToken();

                try self.nodeSetData(&node, .expression_literal_boolean, .{
                    .token = literal,
                });
            },
            .identifier => {
                if (lhs != Ast.NodeIndex.nil and !binary.isBinaryExpression(self.previousTokenTag())) {
                    return self.unexpectedToken();
                }

                node = try self.reserveNode(.expression_literal_number);

                const identifier = try self.expectToken(.identifier);

                try self.nodeSetData(&node, .expression_identifier, .{
                    .token = identifier,
                });
            },
            .keyword_int,
            .keyword_uint,
            .keyword_float,
            .keyword_double,
            .keyword_bool,
            .keyword_vec2,
            .keyword_vec3,
            .keyword_vec4,
            .keyword_ivec2,
            .keyword_ivec3,
            .keyword_ivec4,
            .keyword_uvec2,
            .keyword_uvec3,
            .keyword_uvec4,
            .keyword_bvec2,
            .keyword_bvec3,
            .keyword_bvec4,
            .keyword_dvec2,
            .keyword_dvec3,
            .keyword_dvec4,
            .keyword_mat2,
            .keyword_mat3,
            .keyword_mat4,
            .keyword_mat2x2,
            .keyword_mat3x3,
            .keyword_mat4x4,
            .keyword_mat2x3,
            .keyword_mat3x4,
            .keyword_mat3x2,
            .keyword_mat4x3,
            .keyword_dmat2,
            .keyword_dmat3,
            .keyword_dmat4,
            .keyword_dmat2x2,
            .keyword_dmat3x3,
            .keyword_dmat4x4,
            .keyword_dmat2x3,
            .keyword_dmat3x4,
            .keyword_dmat3x2,
            .keyword_dmat4x3,
            => {
                if (lhs != Ast.NodeIndex.nil and !binary.isBinaryExpression(self.previousTokenTag())) {
                    return self.unexpectedToken();
                }

                node = try self.reserveNode(.expression_identifier);

                const keyword_token: Ast.TokenIndex = try self.nextToken();

                try self.nodeSetData(&node, .expression_identifier, .{
                    .token = keyword_token,
                });
            },
            .left_paren => {
                const open_paren = try self.eatToken(.left_paren);

                if (self.peekTokenTag() == .right_paren) {
                    node = .nil;
                } else {
                    node = try self.parseExpression(.{});
                }

                switch (lhs.tag) {
                    .expression_identifier,
                    => {
                        const identifier_data: Ast.Node.Identifier = self.node_heap.getNodePtrConst(.expression_identifier, lhs.index).*;

                        var sub_node = try self.reserveNode(.expression_binary_proc_call);

                        try self.nodeSetData(&sub_node, .expression_binary_proc_call, .{
                            .op_token = identifier_data.token,
                            .left = lhs,
                            .right = node,
                        });

                        node = sub_node;
                    },
                    else => {},
                }

                if (open_paren) |_| {
                    _ = try self.eatToken(.right_paren);
                }
            },
            .left_bracket => {
                const open_bracket = try self.eatToken(.left_bracket);

                if (self.peekTokenTag() == .right_bracket) {
                    node = .nil;
                } else {
                    node = try self.parseExpression(.{});
                }

                switch (lhs.tag) {
                    .expression_identifier,
                    => {
                        const identifier_data: Ast.Node.Identifier = self.node_heap.getNodePtrConst(.expression_identifier, lhs.index).*;

                        var sub_node = try self.reserveNode(.expression_binary_array_access);

                        try self.nodeSetData(&sub_node, .expression_binary_array_access, .{
                            .op_token = identifier_data.token,
                            .left = lhs,
                            .right = node,
                        });

                        node = sub_node;
                    },
                    else => {},
                }

                if (open_bracket) |_| {
                    _ = try self.eatToken(.right_bracket);
                }
            },
            inline else => |tag| {
                const op_token: Ast.TokenIndex = self.peekToken();

                if (binary.getNodeType(tag)) |binary_node_type| {
                    const prec = binary.getPrecedence(binary_node_type);

                    if (prec <= context.min_precedence) {
                        break;
                    } else {
                        _ = try self.nextToken();

                        const rhs = try self.parseExpression(.{
                            .min_precedence = prec,
                        });

                        var sub_node = try self.reserveNode(binary_node_type);

                        @setEvalBranchQuota(1000000);

                        try self.nodeSetData(&sub_node, binary_node_type, .{
                            .op_token = op_token,
                            .left = lhs,
                            .right = rhs,
                        });

                        node = sub_node;

                        // switch (tag) {
                        // .left_bracket => {
                        // _ = try self.expectToken(.right_bracket);
                        // },
                        // else => {},
                        // }
                    }
                } else {
                    break;
                }
            },
        }

        lhs = node;
    }

    return lhs;
}

pub fn parseTypeExpr(self: *Parser) !Ast.NodeIndex {
    switch (self.peekToken().tag) {
        .keyword_float,
        .keyword_double,
        .keyword_uint,
        .keyword_int,
        .keyword_bool,
        .keyword_vec2,
        .keyword_vec3,
        .keyword_vec4,
        .keyword_ivec2,
        .keyword_ivec3,
        .keyword_ivec4,
        .keyword_uvec2,
        .keyword_uvec3,
        .keyword_uvec4,
        .keyword_bvec2,
        .keyword_bvec3,
        .keyword_bvec4,
        .keyword_dvec2,
        .keyword_dvec3,
        .keyword_dvec4,
        .keyword_mat2x2,
        .keyword_mat3x3,
        .keyword_mat4x4,
        .keyword_mat2x3,
        .keyword_mat3x4,
        .keyword_mat3x2,
        .keyword_mat4x3,
        .keyword_dmat2,
        .keyword_dmat3,
        .keyword_dmat4,
        .keyword_dmat2x2,
        .keyword_dmat3x3,
        .keyword_dmat4x4,
        .keyword_dmat2x3,
        .keyword_dmat3x4,
        .keyword_dmat3x2,
        .keyword_dmat4x3,
        .keyword_void,
        .identifier,
        => {
            var node = try self.reserveNode(.type_expr);
            errdefer self.unreserveNode(node);

            try self.nodeSetData(&node, .type_expr, .{ .token = self.peekToken() });

            _ = try self.nextToken();

            return node;
        },
        else => return self.unexpectedToken(),
    }

    unreachable;
}

pub fn reserveNode(self: *Parser, comptime tag: Ast.Node.Tag) !Ast.NodeIndex {
    const node_index = try self.node_heap.allocateNode(self.gpa, tag);

    return .{
        .tag = tag,
        .index = node_index,
    };
}

pub fn unreserveNode(self: *Parser, node: Ast.NodeIndex) void {
    self.node_heap.freeNode(node);
}

pub fn nodeSetData(
    self: *Parser,
    node: *Ast.NodeIndex,
    comptime Tag: std.meta.Tag(Ast.Node.Data),
    value: std.meta.TagPayload(Ast.Node.Data, Tag),
) !void {
    node.tag = Tag;

    self.node_heap.getNodePtr(Tag, node.index).* = value;
}

pub fn expectToken(self: *Parser, tag: Token.Tag) !Ast.TokenIndex {
    const result_token = try self.eatToken(tag);

    if (result_token == null) {
        try self.errors.append(self.gpa, .{
            .tag = .expected_token,
            .anchor = .{ .token = self.peekToken() },
            .data = .{
                .expected_token = tag,
            },
        });

        return error.ExpectedToken;
    }

    return result_token.?;
}

pub fn unexpectedToken(self: *Parser) anyerror {
    if (self.peekToken().tag != .invalid) self.errors.append(self.gpa, .{
        .tag = .unexpected_token,
        .anchor = .{ .token = self.peekToken() },
    }) catch unreachable;

    return error.UnexpectedToken;
}

pub fn eatToken(self: *Parser, tag: Token.Tag) !?Ast.TokenIndex {
    if (self.peekToken().tag == tag) {
        return try self.nextToken();
    } else {
        return null;
    }
}

pub fn nextToken(self: *Parser) !Ast.TokenIndex {
    //window: previous current next
    //new window: current next next + 1

    const current_window = self.token_window;

    self.token_window[0] = current_window[1];
    self.token_window[1] = current_window[2];

    //TODO: handle this error
    const token = try self.advanceTokenizer();

    self.token_window[2] = token;

    return current_window[1];
}

pub fn previousToken(self: Parser) Ast.TokenIndex {
    return self.lookAheadToken(-1);
}

pub fn previousTokenTag(self: Parser) Token.Tag {
    return self.previousToken().tag;
}

pub fn nextTokenTag(self: *Parser) ?Token.Tag {
    return self.token_tags[self.nextToken() orelse return null];
}

pub fn peekTokenTag(self: Parser) ?Token.Tag {
    return self.token_window[1].tag;
}

pub fn lookAheadTokenTag(self: Parser, comptime amount: comptime_int) Token.Tag {
    const token = self.lookAheadToken(amount);

    return token.tag;
}

pub fn peekToken(self: Parser) Ast.TokenIndex {
    return self.token_window[1];
}

pub fn lookAheadToken(self: Parser, comptime amount: comptime_int) Ast.TokenIndex {
    if (@abs(amount) > 1) {
        @compileError("Token lookahead exhausts the token window buffer!");
    }

    //token_window[1] is the centre (current token)
    return self.token_window[1 + amount];
}

pub fn pushTokenizerState(self: *Parser, new_source_range: Tokenizer.SourceRange, new_define_generation: u32) !void {
    const saved_state = try self.tokenizer_stack.addOne(self.gpa);

    saved_state.define_generation = self.define_generation;
    saved_state.tokenizer = self.tokenizer;

    self.tokenizer = .init(self.source[new_source_range.start..new_source_range.end]);
    self.define_generation = new_define_generation;
}

pub fn popTokenizerState(self: *Parser) !void {
    const old_state = self.tokenizer_stack.pop() orelse return;

    self.define_generation = old_state.define_generation;
    self.tokenizer = old_state.tokenizer;
}

pub fn advanceTokenizer(self: *Parser) anyerror!Ast.TokenIndex {
    while (self.tokenizer.next()) |token| {
        switch (token.tag) {
            .invalid => {
                try self.errors.append(self.gpa, .{
                    .tag = .invalid_token,
                    .anchor = .{ .token = .fromToken(self.source, self.tokenizer.source, token) },
                });

                return .fromToken(self.source, self.tokenizer.source, token);
            },
            .reserved_keyword => {
                try self.errors.append(self.gpa, .{
                    .tag = .reserved_keyword_token,
                    .anchor = .{ .token = .fromToken(self.source, self.tokenizer.source, token) },
                });

                return .fromToken(self.source, self.tokenizer.source, token);
            },
            .directive_version => {
                const string = "__VERSION__";

                const define = self.defines.getOrPut(self.gpa, string) catch unreachable;
                _ = define; // autofix

                const next_token = self.tokenizer.next();
                _ = next_token; // autofix

                //TODO: add expected literal number error here
                //TODO: handle the defining of __VERSION__

                // define.value_ptr.start_token = .fromToken(next_token.?);
            },
            .directive_if,
            .directive_ifdef,
            .directive_ifndef,
            .directive_elif,
            => {
                if (token.tag != .directive_elif) {
                    const identifier_token = self.tokenizer.next() orelse break;
                    const identifier_actual_token: Ast.TokenIndex = .fromToken(self.source, self.tokenizer.source, identifier_token);

                    const condition_string = self.source[identifier_actual_token.string_start .. identifier_actual_token.string_start + identifier_actual_token.string_length];

                    self.directive_if_level += 1;

                    //TODO: handle preprocessor errors

                    switch (token.tag) {
                        .directive_if => {
                            switch (identifier_token.tag) {
                                .identifier => {
                                    const string = condition_string;

                                    const define_source_range, const define_source_generation = self.directiveResolveMacro(
                                        string,
                                    ) orelse @panic("TODO: error message not implemented");
                                    _ = define_source_generation; // autofix

                                    var define_tokenizer: Tokenizer = .init(self.source[define_source_range.start..define_source_range.end]);

                                    //TODO: handle preprocessor expressions
                                    const value_token = define_tokenizer.next().?;

                                    const value = try std.fmt.parseUnsigned(
                                        u64,
                                        self.tokenizer.source[value_token.start..value_token.end],
                                        10,
                                    );
                                    self.directive_if_condition = value != 0;
                                },
                                .literal_number => {
                                    const value = try std.fmt.parseUnsigned(
                                        u64,
                                        self.tokenizer.source[identifier_token.start..identifier_token.end],
                                        10,
                                    );
                                    self.directive_if_condition = value != 0;
                                },
                                else => unreachable,
                            }
                        },
                        .directive_ifdef => {
                            self.directive_if_condition = self.defines.contains(condition_string);
                        },
                        .directive_ifndef => {
                            self.directive_if_condition = !self.defines.contains(condition_string);
                        },
                        else => unreachable,
                    }
                } else {
                    self.directive_if_condition = !self.directive_if_condition;
                }

                if (self.directive_if_condition) continue;

                var if_condition_level: u32 = 1;

                loop_skip: while (true) {
                    if (self.tokenizer.advanceUntilNextDirective() == null) {
                        try self.errors.append(self.gpa, .{
                            .tag = .expected_endif,
                            .anchor = .{ .token = .fromToken(self.source, self.tokenizer.source, token) },
                        });

                        return error.UnexpectedEndif;
                    }

                    //TODO: handle preprocessor errors
                    const directive_token = self.tokenizer.next() orelse break;

                    switch (directive_token.tag) {
                        .directive_if,
                        .directive_ifdef,
                        .directive_ifndef,
                        => {
                            if_condition_level += 1;
                        },
                        .directive_endif,
                        .directive_elif,
                        .directive_end,
                        => {
                            if_condition_level -= 1;

                            if (if_condition_level == 0) {
                                break :loop_skip;
                            }
                        },
                        else => {},
                    }
                }
            },
            .directive_endif => {
                if (self.directive_if_level == 0) {
                    try self.errors.append(self.gpa, .{
                        .tag = .unexpected_endif,
                        .anchor = .{ .token = .fromToken(self.source, self.tokenizer.source, token) },
                    });

                    return error.UnexpectedEndif;
                }

                self.directive_if_level -= 1;
            },
            .directive_define => {
                const identifier_token = self.tokenizer.next() orelse break;

                const string = self.tokenizer.source[identifier_token.start..identifier_token.end];

                self.define_generation += 1;

                const define = self.defines.getOrPut(self.gpa, string) catch unreachable;

                if (!define.found_existing) {
                    define.value_ptr.* = .{};
                }

                //TODO: cache the tokenisation of defines to avoid repeat tokenization, with a heuristic based on text size vs token list size

                //TODO: handle errors

                //keep going until new line
                //TODO: handle line continuation
                const line_range = self.tokenizer.advanceLineRange();

                try define.value_ptr.*.generation_to_definition.put(self.gpa, self.define_generation, line_range);
            },
            .directive_undef => {
                const identifier_token = self.tokenizer.next() orelse break;

                _ = self.defines.remove(self.tokenizer.source[identifier_token.start..identifier_token.end]);
            },
            .directive_error => {
                try self.errors.append(self.gpa, .{
                    .tag = .directive_error,
                    .anchor = .{ .token = .fromToken(self.source, self.tokenizer.source, token) },
                });

                return error.DirectiveError;
            },
            .directive_end => {},
            .identifier => {
                const macro_resolved_range, const macro_resolved_generation = self.directiveResolveMacro(self.tokenizer.source[token.start..token.end]) orelse {
                    return .fromToken(self.source, self.tokenizer.source, token);
                };

                try self.pushTokenizerState(macro_resolved_range, macro_resolved_generation);

                return try self.advanceTokenizer();
            },
            .directive_line,
            .directive_include,
            => {
                try self.errors.append(self.gpa, .{
                    .tag = .unsupported_directive,
                    .anchor = .{ .token = .fromToken(self.source, self.tokenizer.source, token) },
                });
            },
            else => {
                return .fromToken(self.source, self.tokenizer.source, token);
            },
        }
    }

    if (self.tokenizer_stack.items.len > 0) {
        try self.popTokenizerState();

        return try self.advanceTokenizer();
    }

    return .end_of_file;
}

//TODO: make this a distinct enum type?
const DefineGeneration = u32;

fn directiveResolveMacro(
    self: *Parser,
    string: []const u8,
) ?struct { Tokenizer.SourceRange, DefineGeneration } {
    const define = self.defines.get(string) orelse return null;

    var most_recent_define_generation: u32 = 0;
    var most_recent_define_index: ?u32 = 0;

    for (define.generation_to_definition.keys(), 0..) |define_generation, index| {
        if (define_generation <= self.define_generation) {
            most_recent_define_generation = @max(most_recent_define_generation, define_generation);

            if (define_generation >= most_recent_define_generation) {
                most_recent_define_index = @intCast(index);
            }
        }
    }

    if (most_recent_define_index == null) return null;

    return .{ define.generation_to_definition.values()[most_recent_define_index.?], most_recent_define_generation };
}

pub const TokenWindow = [3]Ast.TokenIndex;

pub const Define = struct {
    //TODO: handle multiple files
    generation_to_definition: std.AutoArrayHashMapUnmanaged(u32, Tokenizer.SourceRange) = .{},
};

pub const DefineMap = token_map.Map(Define);

const std = @import("std");
const Parser = @This();
const Ast = @import("Ast.zig");
const token_map = @import("token_map.zig");
const Token = Tokenizer.Token;
const Tokenizer = @import("Tokenizer.zig");
