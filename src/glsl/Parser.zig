//! Implements the syntactic analysis stage of the frontend

allocator: std.mem.Allocator,
source: []const u8,
token_tags: []const Token.Tag,
token_starts: []const u32,
token_ends: []const u32,
token_index: u32,
node_context_stack: std.ArrayList(struct {
    saved_token_index: u32,
    saved_error_index: u32,
}),
errors: std.ArrayList(Ast.Error),
node_heap: Ast.NodeHeap = .{},
root_decls: []Ast.NodeIndex,

pub fn init(
    allocator: std.mem.Allocator,
    source: []const u8,
    tokens: Ast.TokenList.Slice,
) Parser {
    return .{
        .allocator = allocator,
        .source = source,
        .token_tags = tokens.items(.tag),
        .token_starts = tokens.items(.start),
        .token_ends = tokens.items(.end),
        .token_index = 0,
        .node_context_stack = .{},
        .errors = .{},
        .root_decls = &.{},
    };
}

pub fn deinit(self: *Parser) void {
    defer self.* = undefined;
    defer self.errors.deinit(self.allocator);
    defer self.node_context_stack.deinit(self.allocator);
    // defer self.node_heap.deinit(self.allocator);
}

///Root parse node
pub fn parse(self: *Parser) !void {
    var state: enum {
        start,
        directive,
    } = .start;

    var root_nodes: std.ArrayList(Ast.NodeIndex) = .{};
    defer root_nodes.deinit(self.allocator);

    _ = try self.expectToken(.directive_version);
    _ = try self.expectToken(.literal_number);

    while (self.token_index < self.token_tags.len) {
        switch (state) {
            .start => switch (self.token_tags[self.token_index]) {
                .directive_define,
                .directive_undef,
                .directive_if,
                .directive_ifdef,
                .directive_ifndef,
                .directive_else,
                .directive_elif,
                .directive_endif,
                .directive_error,
                .directive_pragma,
                .directive_extension,
                .directive_version,
                .directive_line,
                => {
                    state = .directive;
                    _ = self.nextToken();
                },
                .keyword_struct => {
                    const struct_def = try self.parseStruct();

                    try root_nodes.append(self.allocator, struct_def);
                },
                .keyword_void,
                .keyword_double,
                .keyword_float,
                .keyword_int,
                .keyword_uint,
                .keyword_bool,
                .identifier,
                => {
                    const proc = try self.parseProcedure();

                    try root_nodes.append(self.allocator, proc);
                },
                else => {
                    _ = self.nextToken();
                },
            },
            .directive => switch (self.token_tags[self.token_index]) {
                .directive_end => {
                    state = .start;
                    _ = self.nextToken();
                },
                else => {
                    _ = self.nextToken();
                },
            },
        }
    }

    self.root_decls = try root_nodes.toOwnedSlice(self.allocator);
}

pub fn parseStruct(self: *Parser) !Ast.NodeIndex {
    var node_index = try self.reserveNode(.struct_definition);
    errdefer self.unreserveNode(node_index);

    _ = try self.expectToken(.keyword_struct);

    const struct_name_identifier = try self.expectToken(.identifier);

    _ = try self.expectToken(.left_brace);

    var field_nodes: std.ArrayList(Ast.NodeIndex) = .{};
    defer field_nodes.deinit(self.allocator);

    while (self.peekTokenTag().? != .right_brace) {
        const field_type = try self.parseTypeExpr();

        const field_name = try self.expectToken(.identifier);

        var struct_field_node = try self.reserveNode(.struct_field);

        try self.nodeSetData(&struct_field_node, .struct_field, .{
            .name = field_name,
            .type_expr = field_type,
        });

        try field_nodes.append(self.allocator, struct_field_node);

        _ = try self.expectToken(.semicolon);
    }

    try self.nodeSetData(&node_index, .struct_definition, .{
        .name = struct_name_identifier,
        .fields = try self.allocator.dupe(Ast.NodeIndex, field_nodes.items),
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
    defer param_nodes.deinit(self.allocator);

    while (self.peekTokenTag().? != .right_paren) {
        const param = try self.parseParam();

        try param_nodes.append(self.allocator, param);

        if (self.lookAheadTokenTag(1).? != .right_paren) {
            _ = self.eatToken(.comma);
        }
    }

    if (param_nodes.items.len == 0) {
        return Ast.NodeIndex.nil;
    }

    try self.nodeSetData(&node, .param_list, .{
        .params = try self.node_heap.allocateDupe(self.allocator, Ast.NodeIndex, param_nodes.items),
    });

    return node;
}

pub fn parseParam(self: *Parser) !Ast.NodeIndex {
    var node = try self.reserveNode(.param_expr);
    errdefer self.unreserveNode(node);

    const qualifier: Token.Tag = self.eatVariableQualifier();

    const type_expr = try self.parseTypeExpr();
    const param_identifier = try self.expectToken(.identifier);

    try self.nodeSetData(&node, .param_expr, .{
        .type_expr = type_expr,
        .name = param_identifier,
        .qualifier = qualifier,
    });

    return node;
}

pub fn eatVariableQualifier(self: *Parser) Token.Tag {
    var qualifier: Token.Tag = .keyword_in;

    const first_in_qualifier = self.eatToken(.keyword_in);

    switch (self.peekTokenTag().?) {
        .keyword_inout,
        .keyword_out,
        .keyword_in,
        .keyword_const,
        => |tag| {
            qualifier = tag;

            _ = self.nextToken();
        },
        else => {},
    }

    if (first_in_qualifier == null) {
        _ = self.eatToken(.keyword_in);
    }

    return qualifier;
}

pub fn parseStatement(self: *Parser) !Ast.NodeIndex {
    switch (self.peekTokenTag().?) {
        .left_brace => {
            var node = try self.reserveNode(.statement_block);
            errdefer self.unreserveNode(node);

            _ = self.nextToken();

            var statements: std.ArrayList(Ast.NodeIndex) = .{};
            defer statements.deinit(self.allocator);

            while (self.peekTokenTag().? != .right_brace) {
                const statement = try self.parseStatement();

                if (statement == Ast.NodeIndex.nil) continue;

                try statements.append(self.allocator, statement);
            }

            _ = try self.expectToken(.right_brace);

            try self.nodeSetData(&node, .statement_block, .{
                .statements = try self.node_heap.allocateDupe(self.allocator, Ast.NodeIndex, statements.items),
            });

            return node;
        },
        .keyword_const,
        .keyword_float,
        .keyword_uint,
        .keyword_int,
        .keyword_bool,
        .keyword_void,
        => {
            var node = try self.reserveNode(.statement_var_init);
            errdefer self.unreserveNode(node);

            const qualifier = self.eatVariableQualifier();

            const type_expr = try self.parseTypeExpr();

            const variable_name = try self.expectToken(.identifier);

            if (self.eatToken(.equals) != null) {
                const expression = try self.parseExpression(.{});

                try self.nodeSetData(&node, .statement_var_init, .{
                    .type_expr = type_expr,
                    .identifier = variable_name,
                    .expression = expression,
                    .qualifier = qualifier,
                });
            } else {
                try self.nodeSetData(&node, .statement_var_init, .{
                    .type_expr = type_expr,
                    .identifier = variable_name,
                    .qualifier = qualifier,
                    .expression = Ast.NodeIndex.nil,
                });
            }

            return node;
        },
        .identifier,
        .literal_number,
        => {
            defer _ = self.eatToken(.semicolon);

            return self.parseExpression(.{});
        },
        .literal_string,
        => {
            //TODO: what to do with string literals
            _ = self.nextToken();

            std.log.err("String literals are not supported yet", .{});

            return Ast.NodeIndex.nil;
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

            const else_keyword = self.eatToken(.keyword_else);

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
            const keyword_return = self.nextToken().?;

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
            _ = self.nextToken();
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
                    .expression_binary_mul,
                    .expression_binary_div,
                    => 6,
                    .expression_binary_add,
                    .expression_binary_sub,
                    => 5,
                    .expression_binary_eql,
                    .expression_binary_neql,
                    => 4,
                    .expression_binary_lt,
                    .expression_binary_gt,
                    .expression_binary_leql,
                    .expression_binary_geql,
                    => 3,
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

                const literal = self.nextToken().?;

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

                const literal = self.nextToken().?;

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
            .left_paren => {
                const open_paren = self.eatToken(.left_paren);
                defer if (open_paren) |_| {
                    _ = self.eatToken(.right_paren);
                };

                node = try self.parseExpression(.{});

                if (lhs.tag == .expression_identifier) {
                    const identifier_data: Ast.Node.Identifier = self.node_heap.getNodePtrConst(.expression_identifier, lhs.index).*;

                    var sub_node = try self.reserveNode(.expression_binary_proc_call);

                    try self.nodeSetData(&sub_node, .expression_binary_proc_call, .{
                        .op_token = identifier_data.token,
                        .left = lhs,
                        .right = node,
                    });

                    node = sub_node;
                }
            },
            inline else => |tag| {
                const op_token = self.token_index;

                defer switch (tag) {
                    .left_bracket => {
                        _ = self.eatToken(.right_bracket);
                    },
                    else => {},
                };

                if (binary.getNodeType(tag)) |binary_node_type| {
                    const prec = binary.getPrecedence(binary_node_type);

                    if (prec <= context.min_precedence) {
                        break;
                    } else {
                        _ = self.nextToken();

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
    switch (self.token_tags[self.token_index]) {
        .keyword_void,
        .keyword_int,
        .keyword_uint,
        .keyword_float,
        .keyword_double,
        .keyword_vec2,
        .keyword_vec3,
        .keyword_vec4,
        .keyword_bool,
        .identifier,
        => {
            var node = try self.reserveNode(.type_expr);
            errdefer self.unreserveNode(node);

            defer self.token_index += 1;

            try self.nodeSetData(&node, .type_expr, .{ .token = self.token_index });

            return node;
        },
        else => return self.unexpectedToken(),
    }

    unreachable;
}

pub fn reserveNode(self: *Parser, comptime tag: Ast.Node.Tag) !Ast.NodeIndex {
    try self.node_context_stack.append(self.allocator, .{
        .saved_token_index = self.token_index,
        .saved_error_index = @intCast(self.errors.items.len),
    });

    const node_index = try self.node_heap.allocateNode(self.allocator, tag);

    return .{
        .tag = tag,
        .index = node_index,
    };
}

pub fn unreserveNode(self: *Parser, node: Ast.NodeIndex) void {
    self.node_heap.freeNode(node);

    const context = self.node_context_stack.pop().?;

    self.token_index = context.saved_token_index;
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

pub fn tokenIndexString(self: Parser, token_index: u32) []const u8 {
    return self.tokenString(.{
        .start = self.token_starts[token_index],
        .end = self.token_ends[token_index],
        .tag = self.token_tags[token_index],
    });
}

pub fn tokenString(self: Parser, token: Token) []const u8 {
    return self.source[token.start..token.end];
}

pub fn expectToken(self: *Parser, tag: Token.Tag) !u32 {
    errdefer self.errors.append(self.allocator, .{
        .tag = .expected_token,
        .token = self.token_index,
        .data = .{
            .expected_token = tag,
        },
    }) catch unreachable;

    return self.eatToken(tag) orelse error.ExpectedToken;
}

pub fn unexpectedToken(self: *Parser) anyerror {
    if (self.token_tags[self.token_index] != .invalid) self.errors.append(self.allocator, .{
        .tag = .unexpected_token,
        .token = self.token_index,
    }) catch unreachable;

    return error.UnexpectedToken;
}

pub fn eatToken(self: *Parser, tag: Token.Tag) ?u32 {
    if (self.token_index < self.token_tags.len and self.peekTokenTag() != null and self.peekTokenTag() == tag) {
        return self.nextToken();
    } else {
        return null;
    }
}

pub fn nextToken(self: *Parser) ?u32 {
    const result = self.peekToken();

    self.token_index += 1;

    return result;
}

pub fn previousToken(self: Parser) u32 {
    return self.token_index - 1;
}

pub fn previousTokenTag(self: Parser) Token.Tag {
    return self.token_tags[self.previousToken()];
}

pub fn nextTokenTag(self: *Parser) ?Token.Tag {
    return self.token_tags[self.nextToken() orelse return null];
}

pub fn peekTokenTag(self: Parser) ?Token.Tag {
    return self.lookAheadTokenTag(0);
}

pub fn lookAheadTokenTag(self: Parser, amount: u32) ?Token.Tag {
    return self.token_tags[self.lookAheadToken(amount) orelse return null];
}

//TODO: support preprocessor directives inside function bodies by modifying this to allow that
pub fn peekToken(self: Parser) ?u32 {
    return self.lookAheadToken(0);
}

pub fn lookAheadToken(self: Parser, amount: u32) ?u32 {
    const result = self.token_index + amount;

    if (result >= self.token_tags.len) {
        return null;
    }

    return result;
}

const std = @import("std");
const Parser = @This();
const Ast = @import("Ast.zig");
const Token = @import("Tokenizer.zig").Token;
