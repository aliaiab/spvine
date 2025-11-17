//! Implements the syntactic analysis stage of the glsl frontend

gpa: std.mem.Allocator,
ast_node_arena: std.mem.Allocator,
scratch_arena: std.mem.Allocator,
sources: std.ArrayList([]const u8),
///Maps from paths to source indices
source_map: token_map.ArrayMap(u32) = .{},
///An index of the current source file
source_index: usize = 0,
tokenizer: Tokenizer,
///Stores a centered slice of the token stream
///The centre (token_window[1]) is the current token
token_window: TokenWindow = undefined,
defines: DefineMap = .{},
///This is the generation of #define we're currently on
define_generation: u32 = 0,
directive_if_level: u32 = 0,
directive_if_condition: bool = false,
tokenizer_stack: std.ArrayList(struct {
    //This may be able to be inferred from the token window (bar the retained state within Tokenizer)
    tokenizer: Tokenizer,
    define_generation: u32,
    source_index: usize,
}) = .{},
errors: std.ArrayList(Ast.Error),
root_decls: []Ast.NodePointer,

pub fn init(
    gpa: std.mem.Allocator,
    ast_node_arena: std.mem.Allocator,
    scratch_arena: std.mem.Allocator,
    source: []const u8,
    source_name: []const u8,
) !Parser {
    var sources: std.ArrayList([]const u8) = .{};

    try sources.append(gpa, source);

    var source_map: token_map.ArrayMap(u32) = .{};

    try source_map.put(gpa, source_name, 0);

    return .{
        .gpa = gpa,
        .ast_node_arena = ast_node_arena,
        .scratch_arena = scratch_arena,
        .sources = sources,
        .source_map = source_map,
        .tokenizer = Tokenizer.init(source),
        .errors = .{},
        .root_decls = &.{},
    };
}

pub fn deinit(self: *Parser) void {
    self.errors.deinit(self.gpa);
    self.* = undefined;
}

///Root parse node
pub fn parse(self: *Parser) !void {
    var root_nodes: std.ArrayList(Ast.NodePointer) = .{};
    defer root_nodes.deinit(self.gpa);

    //Init the token window
    self.token_window[0] = .nil;

    for (self.token_window[1..]) |*token| {
        token.* = try self.advanceTokenizer();
    }

    while (self.peekToken().tag != .end_of_file) {
        switch (self.peekToken().tag) {
            .keyword_struct => {
                const struct_def = try self.parseStruct();

                try root_nodes.append(self.gpa, struct_def);
            },
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

pub fn parseStruct(self: *Parser) !Ast.NodePointer {
    var struct_node = try self.allocateNode(.struct_definition);

    _ = try self.expectToken(.keyword_struct);

    const struct_name_identifier = try self.expectToken(.identifier);

    _ = try self.expectToken(.left_brace);

    var field_nodes: std.ArrayList(Ast.NodeRelativePointer) = .{};
    defer field_nodes.deinit(self.scratch_arena);

    while (self.peekTokenTag().? != .right_brace) {
        const field_type = try self.parseTypeExpr();

        const field_name = try self.expectToken(.identifier);

        var struct_field_node = try self.allocateNode(.struct_field);

        struct_field_node.data(Ast.Node.StructField).* = .{
            .name = field_name,
            .type_expr = .relativeTo(struct_field_node, field_type),
        };

        try field_nodes.append(self.scratch_arena, .relativeTo(struct_node, struct_field_node));

        _ = try self.expectToken(.semicolon);
    }

    struct_node.data(Ast.Node.StructDefinition).* = .{
        .name = struct_name_identifier,
        .fields = try self.ast_node_arena.dupe(Ast.NodeRelativePointer, field_nodes.items),
    };

    _ = try self.expectToken(.right_brace);
    _ = try self.expectToken(.semicolon);

    return struct_node;
}

pub fn parseProcedure(self: *Parser) !Ast.NodePointer {
    var procedure_node = try self.allocateNode(.procedure);

    const type_expr = try self.parseTypeExpr();

    const identifier = try self.expectToken(.identifier);

    _ = try self.expectToken(.left_paren);

    const param_list = try self.parseParamList();

    _ = try self.expectToken(.right_paren);

    var body: Ast.NodePointer = .nil;

    if (self.peekTokenTag() == .left_brace) {
        body = try self.parseStatement();
    } else {
        _ = try self.expectToken(.semicolon);
    }

    procedure_node.data(Ast.Node.Procedure).* = .{
        .return_type = .relativeTo(procedure_node, type_expr),
        .name = identifier,
        .param_list = .relativeTo(procedure_node, param_list),
        .body = .relativeTo(procedure_node, body),
    };

    return procedure_node;
}

pub fn parseParamList(self: *Parser) !Ast.NodePointer {
    var param_list_node = try self.allocateNode(.param_list);

    var param_nodes: std.ArrayList(Ast.NodeRelativePointer) = .{};
    defer param_nodes.deinit(self.gpa);

    while (self.peekTokenTag().? != .right_paren) {
        const param = try self.parseParam();

        try param_nodes.append(self.gpa, .relativeTo(param_list_node, param));

        if (self.lookAheadTokenTag(1) != .right_paren) {
            _ = try self.eatToken(.comma);
        }
    }

    param_list_node.data(Ast.Node.ParamList).* = .{
        .params = try self.ast_node_arena.dupe(Ast.NodeRelativePointer, param_nodes.items),
    };

    return param_list_node;
}

pub fn parseParam(self: *Parser) !Ast.NodePointer {
    var param_expr_node = try self.allocateNode(.param_expr);

    const qualifier: Ast.TokenIndex = try self.parseVariableQualifier();

    const type_expr = try self.parseTypeExpr();
    const param_identifier = try self.expectToken(.identifier);

    param_expr_node.data(Ast.Node.ParamExpr).* = .{
        .type_expr = .relativeTo(param_expr_node, type_expr),
        .name = param_identifier,
        .qualifier = qualifier,
    };

    return param_expr_node;
}

pub fn parseVariableQualifier(self: *Parser) !Ast.TokenIndex {
    var qualifier: Ast.TokenIndex = .nil;

    const first_in_qualifier = try self.eatToken(.keyword_in);

    switch (self.peekTokenTag().?) {
        .keyword_inout,
        .keyword_out,
        .keyword_in,
        .keyword_const,
        => {
            qualifier = try self.nextToken();
        },
        else => {},
    }

    if (first_in_qualifier == null) {
        _ = try self.eatToken(.keyword_in);
    }

    return qualifier;
}

pub fn parseStatement(self: *Parser) !Ast.NodePointer {
    switch (self.peekTokenTag().?) {
        .left_brace => {
            var statement_block_node = try self.allocateNode(.statement_block);

            _ = try self.nextToken();

            var statements: std.ArrayList(Ast.NodeRelativePointer) = .{};
            defer statements.deinit(self.scratch_arena);

            while (self.peekTokenTag().? != .right_brace) {
                const statement = try self.parseStatement();

                if (statement == Ast.NodePointer.nil) continue;

                try statements.append(self.scratch_arena, .relativeTo(statement_block_node, statement));
            }

            _ = try self.expectToken(.right_brace);

            statement_block_node.data(Ast.Node.StatementBlock).* = .{
                .statements = try self.ast_node_arena.dupe(Ast.NodeRelativePointer, statements.items),
            };

            return statement_block_node;
        },
        .left_paren => {
            return self.parseExpression(.{});
        },
        .keyword_const,
        .identifier,
        => |token_tag| {
            if (token_tag == .identifier) {
                const next_token = self.lookAheadTokenTag(1);

                if (next_token != .end_of_file) {
                    if (next_token != .identifier) {
                        const expr = try self.parseExpression(.{});

                        _ = try self.eatToken(.semicolon);

                        return expr;
                    }
                }
            }

            var var_init_node = try self.allocateNode(.statement_var_init);

            const qualifier = try self.parseVariableQualifier();

            const type_expr = try self.parseTypeExpr();

            const variable_name = try self.expectToken(.identifier);

            const array_left_bracket = try self.eatToken(.left_bracket);

            var array_length_specifier: Ast.NodePointer = .nil;

            if (array_left_bracket != null) {
                const array_length_expr = try self.parseExpression(.{});

                array_length_specifier = array_length_expr;

                _ = try self.expectToken(.right_bracket);
            }

            var expression: Ast.NodePointer = .nil;

            if (try self.eatToken(.equals) != null) {
                expression = try self.parseExpression(.{});
            }

            var_init_node.data(Ast.Node.StatementVarInit).* = .{
                .type_expr = .relativeTo(var_init_node, type_expr),
                .array_length_specifier = .relativeTo(var_init_node, array_length_specifier),
                .identifier = variable_name,
                .expression = .relativeTo(var_init_node, expression),
                .qualifier = qualifier,
            };

            return var_init_node;
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

            var if_node = try self.allocateNode(.statement_if);

            _ = try self.expectToken(.left_paren);

            const cond_expr = try self.parseExpression(.{});

            _ = try self.expectToken(.right_paren);

            const taken_statment = try self.parseStatement();

            var not_taken_statment = Ast.NodePointer.nil;

            const else_keyword = try self.eatToken(.keyword_else);

            if (else_keyword) |_| {
                not_taken_statment = try self.parseStatement();
            }

            if_node.data(Ast.Node.StatementIf).* = .{
                .if_token = if_token,
                .condition_expression = .relativeTo(if_node, cond_expr),
                .taken_statement = .relativeTo(if_node, taken_statment),
                .not_taken_statement = .relativeTo(if_node, not_taken_statment),
            };

            return if_node;
        },
        .keyword_return => {
            const keyword_return = try self.nextToken();

            var return_node = try self.allocateNode(.statement_return);

            var expression: Ast.NodePointer = .nil;

            if (self.peekTokenTag().? != .semicolon) {
                expression = try self.parseExpression(.{});
            }

            _ = try self.expectToken(.semicolon);

            return_node.data(Ast.Node.StatementReturn).* = .{
                .expression = .relativeTo(return_node, expression),
                .return_token = keyword_return,
            };

            return return_node;
        },
        .semicolon => {
            _ = try self.nextToken();
            return .nil;
        },
        else => return self.unexpectedToken(),
    }

    return .nil;
}

pub fn parseExpression(
    self: *Parser,
    context: struct {
        min_precedence: i32 = std.math.minInt(i32),
        left: Ast.NodePointer = Ast.NodePointer.nil,
    },
) anyerror!Ast.NodePointer {
    var lhs = context.left;

    while (true) {
        var node = Ast.NodePointer.nil;

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
                if (lhs != Ast.NodePointer.nil and !binary.isBinaryExpression(self.previousTokenTag())) {
                    return self.unexpectedToken();
                }

                node = try self.allocateNode(.expression_literal_number);

                const literal = try self.nextToken();

                node.data(Ast.Node.ExpressionLiteralNumber).* = .{
                    .token = literal,
                };
            },
            .keyword_true,
            .keyword_false,
            => {
                if (lhs != Ast.NodePointer.nil and !binary.isBinaryExpression(self.previousTokenTag())) {
                    return self.unexpectedToken();
                }

                node = try self.allocateNode(.expression_literal_boolean);

                const literal = try self.nextToken();

                node.data(Ast.Node.LiteralBoolean).* = .{
                    .token = literal,
                };
            },
            .identifier => {
                if (lhs != Ast.NodePointer.nil and !binary.isBinaryExpression(self.previousTokenTag())) {
                    return self.unexpectedToken();
                }

                node = try self.allocateNode(.expression_identifier);

                const identifier = try self.expectToken(.identifier);

                node.data(Ast.Node.Identifier).* = .{
                    .token = identifier,
                };
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
                        const identifier_data = lhs.data(Ast.Node.Identifier).*;

                        var sub_node = try self.allocateNode(.expression_binary_proc_call);

                        sub_node.data(Ast.Node.BinaryExpression).* = .{
                            .op_token = identifier_data.token,
                            .left = .relativeTo(sub_node, lhs),
                            .right = .relativeTo(sub_node, node),
                        };

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
                        const identifier_data = lhs.data(Ast.Node.Identifier);

                        var sub_node = try self.allocateNode(.expression_binary_array_access);

                        sub_node.data(Ast.Node.BinaryExpression).* = .{
                            .op_token = identifier_data.token,
                            .left = .relativeTo(sub_node, lhs),
                            .right = .relativeTo(sub_node, node),
                        };

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

                        var sub_node = try self.allocateNode(binary_node_type);

                        @setEvalBranchQuota(1000000);

                        sub_node.data(Ast.Node.BinaryExpression).* = .{
                            .op_token = op_token,
                            .left = .relativeTo(sub_node, lhs),
                            .right = .relativeTo(sub_node, rhs),
                        };

                        node = sub_node;
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

pub fn parseTypeExpr(self: *Parser) !Ast.NodePointer {
    switch (self.peekToken().tag) {
        .identifier,
        => {
            var node = try self.allocateNode(.type_expr);

            node.data(Ast.Node.TypeExpr).* = .{ .token = self.peekToken() };

            _ = try self.nextToken();

            return node;
        },
        else => return self.unexpectedToken(),
    }

    unreachable;
}

pub fn allocateNode(self: *Parser, comptime tag: Ast.Node.Tag) !Ast.NodePointer {
    const NodeData = std.meta.TagPayload(Ast.Node.Data, tag);

    const ptr = try self.ast_node_arena.create(NodeData);

    return .{
        .tag = tag,
        .data_ptr = @intCast(@intFromPtr(ptr)),
    };
}

pub fn nodeSetData(
    self: *Parser,
    node: *Ast.NodeRelativePointer,
    comptime Tag: std.meta.Tag(Ast.Node.Data),
    value: std.meta.TagPayload(Ast.Node.Data, Tag),
) !void {
    node.tag = Tag;

    self.node_heap.getNodePtr(Tag, node.relative_ptr).* = value;
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

///Advances the token window forward, returning the next logical token
pub fn nextToken(self: *Parser) !Ast.TokenIndex {
    const current_window = self.token_window;

    for (0..current_window.len - 1) |i| {
        self.token_window[i] = current_window[1 + i];
    }

    const token = try self.advanceTokenizer();

    self.token_window[current_window.len - 1] = token;

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

pub fn tokenString(self: *Parser, token_index: Ast.TokenIndex) []const u8 {
    const token_start = token_index.string_start;
    const token_end = token_index.string_start + token_index.string_length;

    const source = self.sources.items[token_index.file_index];

    return source[token_start..token_end];
}

pub fn tokenizerConsume(self: *Parser) !Ast.TokenIndex {
    const token = self.tokenizer.next() orelse return self.unexpectedToken();

    return .fromToken(
        self.source_index,
        self.sources.items[self.source_index],
        self.tokenizer.source,
        token,
    );
}

fn pushTokenizerState(self: *Parser, new_source_range: Ast.SourceStringRange, new_define_generation: u32) !void {
    const saved_state = try self.tokenizer_stack.addOne(self.gpa);

    saved_state.define_generation = self.define_generation;
    saved_state.tokenizer = self.tokenizer;
    saved_state.source_index = self.source_index;

    self.tokenizer = .init(self.sources.items[new_source_range.file_index][new_source_range.start..new_source_range.end]);
    self.define_generation = new_define_generation;
    self.source_index = new_source_range.file_index;
}

fn popTokenizerState(self: *Parser) !void {
    const old_state = self.tokenizer_stack.pop() orelse return;

    self.define_generation = old_state.define_generation;
    self.tokenizer = old_state.tokenizer;
    self.source_index = old_state.source_index;
}

fn advanceTokenizer(self: *Parser) !Ast.TokenIndex {
    while (true) {
        while (self.tokenizer.next()) |token| {
            switch (token.tag) {
                .invalid => {
                    try self.errors.append(self.gpa, .{
                        .tag = .invalid_token,
                        .anchor = .{ .token = .fromToken(self.source_index, self.sources.items[self.source_index], self.tokenizer.source, token) },
                    });

                    return .fromToken(self.source_index, self.sources.items[self.source_index], self.tokenizer.source, token);
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
                .directive_include => {
                    const string_token = self.tokenizer.next() orelse {
                        @panic("Expected string literal");
                    };

                    std.debug.assert(string_token.tag == .literal_string);

                    const dir_name = std.fs.path.dirname(self.source_map.keys()[0]).?;

                    const include_path: []const u8 = self.sources.items[self.source_index][string_token.start + 1 .. string_token.end - 1];

                    const actual_path = try std.fs.path.join(self.gpa, &.{ dir_name, include_path });

                    const included_source = try std.fs.cwd().readFileAlloc(
                        self.gpa,
                        actual_path,
                        std.math.maxInt(u32),
                    );

                    const include_index = self.sources.items.len;

                    try self.sources.append(self.gpa, included_source);

                    try self.source_map.put(self.gpa, actual_path, @intCast(include_index));

                    try self.pushTokenizerState(
                        .{
                            .file_index = @intCast(include_index),
                            .start = 0,
                            .end = @intCast(included_source.len),
                        },
                        self.define_generation,
                    );
                },
                .directive_if,
                .directive_ifdef,
                .directive_ifndef,
                .directive_elif,
                => {
                    if (token.tag != .directive_elif) {
                        self.directive_if_level += 1;

                        switch (token.tag) {
                            .directive_if => {
                                self.directive_if_condition = try self.directiveParseIfCondition() != 0;
                            },
                            .directive_ifdef => {
                                const identifier_token = self.tokenizer.next() orelse break;

                                const identifier_actual_token: Ast.TokenIndex = .fromToken(
                                    self.source_index,
                                    self.sources.items[self.source_index],
                                    self.tokenizer.source,
                                    identifier_token,
                                );

                                const condition_string = self.sources.items[self.source_index][identifier_actual_token.string_start .. identifier_actual_token.string_start + identifier_actual_token.string_length];

                                self.directive_if_condition = self.defines.contains(condition_string);
                            },
                            .directive_ifndef => {
                                const identifier_token = self.tokenizer.next() orelse break;

                                const identifier_actual_token: Ast.TokenIndex = .fromToken(
                                    self.source_index,
                                    self.sources.items[self.source_index],
                                    self.tokenizer.source,
                                    identifier_token,
                                );

                                const condition_string = self.sources.items[self.source_index][identifier_actual_token.string_start .. identifier_actual_token.string_start + identifier_actual_token.string_length];

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
                                .anchor = .{ .token = .fromToken(
                                    self.source_index,
                                    self.sources.items[self.source_index],
                                    self.tokenizer.source,
                                    token,
                                ) },
                            });

                            return error.UnexpectedEndif;
                        }

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
                            .anchor = .{ .token = .fromToken(self.source_index, self.sources.items[self.source_index], self.tokenizer.source, token) },
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

                    try define.value_ptr.*.generation_to_definition.put(
                        self.gpa,
                        self.define_generation,
                        .{
                            .file_index = self.source_index,
                            .start = line_range.start,
                            .end = line_range.end,
                        },
                    );
                },
                .directive_undef => {
                    const identifier_token = self.tokenizer.next() orelse break;

                    _ = self.defines.remove(self.tokenizer.source[identifier_token.start..identifier_token.end]);
                },
                .directive_error => {
                    try self.errors.append(self.gpa, .{
                        .tag = .directive_error,
                        .anchor = .{ .token = .fromToken(
                            self.source_index,
                            self.sources.items[self.source_index],
                            self.tokenizer.source,
                            token,
                        ) },
                    });

                    return error.DirectiveError;
                },
                .directive_end => {},
                .identifier => {
                    const macro_resolved_range, const macro_resolved_generation = self.directiveResolveMacro(self.tokenizer.source[token.start..token.end]) orelse {
                        return .fromToken(
                            self.source_index,
                            self.sources.items[self.source_index],
                            self.tokenizer.source,
                            token,
                        );
                    };

                    try self.pushTokenizerState(macro_resolved_range, macro_resolved_generation);

                    continue;
                },
                .directive_line,
                => {
                    try self.errors.append(self.gpa, .{
                        .tag = .unsupported_directive,
                        .anchor = .{ .token = .fromToken(
                            self.source_index,
                            self.sources.items[self.source_index],
                            self.tokenizer.source,
                            token,
                        ) },
                    });
                },
                else => {
                    return .fromToken(
                        self.source_index,
                        self.sources.items[self.source_index],
                        self.tokenizer.source,
                        token,
                    );
                },
            }
        }

        if (self.tokenizer_stack.items.len == 0) {
            break;
        }

        try self.popTokenizerState();
    }

    return .end_of_file;
}

fn directiveParseIfCondition(
    self: *Parser,
) !i64 {
    const token = try self.tokenizerConsume();

    switch (token.tag) {
        .literal_number => {
            const value = try std.fmt.parseInt(
                i64,
                self.tokenString(token),
                10,
            );

            return value;
        },
        .identifier => {
            if (!std.mem.eql(u8, self.tokenString(token), "defined")) {
                const range, const gen = self.directiveResolveMacro(self.tokenString(token)) orelse unreachable;
                _ = gen; // autofix
                _ = range; // autofix

                //TODO: handle macro expansion in #if properly
                unreachable;
            }

            var left_paren_or_ident = try self.tokenizerConsume();

            if (left_paren_or_ident.tag == .left_paren) {
                left_paren_or_ident = try self.tokenizerConsume();
            }

            const identifier = left_paren_or_ident;

            const maybe_define = self.defines.get(self.tokenString(identifier));

            const condition = maybe_define != null;

            if (left_paren_or_ident.tag == .left_paren) {
                _ = try self.tokenizerConsume();
            }

            return @intFromBool(condition);
        },
        else => |tag| @panic(@tagName(tag)),
    }
}

const DefineGeneration = u32;

fn directiveResolveMacro(
    self: *Parser,
    string: []const u8,
) ?struct { Ast.SourceStringRange, DefineGeneration } {
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
    generation_to_definition: std.AutoArrayHashMapUnmanaged(u32, Ast.SourceStringRange) = .{},
};

pub const DefineMap = token_map.Map(Define);

const std = @import("std");
const Parser = @This();
const Ast = @import("Ast.zig");
const token_map = @import("token_map.zig");
const Token = Tokenizer.Token;
const Tokenizer = @import("Tokenizer.zig");
