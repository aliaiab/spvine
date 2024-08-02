//! Implements the syntactic analysis stage of the frontend

allocator: std.mem.Allocator,
source: []const u8,
token_tags: []const Token.Tag,
token_starts: []const u32,
token_ends: []const u32,
token_index: u32,
node_context_stack: std.ArrayListUnmanaged(struct {
    saved_token_index: u32,
    saved_error_index: u32,
}),
errors: std.ArrayListUnmanaged(Ast.Error),
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
    std.log.info("begin: {s}", .{@src().fn_name});
    defer std.log.info("end: {s}", .{@src().fn_name});

    var state: enum {
        start,
        directive,
    } = .start;

    var root_nodes: std.ArrayListUnmanaged(Ast.NodeIndex) = .{};
    defer root_nodes.deinit(self.allocator);

    // _ = try self.expectToken(.directive_version);
    // _ = try self.expectToken(.literal_number);

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
                    _ = try self.parseStruct();
                },
                .keyword_void,
                .keyword_double,
                .keyword_float,
                .keyword_int,
                .keyword_uint,
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
    std.log.info("begin: {s}", .{@src().fn_name});
    defer std.log.info("end: {s}", .{@src().fn_name});

    const node_index = try self.reserveNode(.procedure);
    errdefer self.unreserveNode(node_index);

    const struct_keyword = try self.expectToken(.keyword_struct);
    _ = struct_keyword; // autofix

    const struct_name_identifier = try self.expectToken(.identifier);
    _ = struct_name_identifier; // autofix

    _ = try self.expectToken(.left_brace);

    while (self.peekTokenTag().? != .right_brace) {
        const field_type = try self.parseTypeExpr();
        _ = field_type; // autofix

        const field_name = try self.expectToken(.identifier);
        _ = field_name; // autofix

        _ = try self.expectToken(.semicolon);
    }

    _ = try self.expectToken(.right_brace);
    _ = try self.expectToken(.semicolon);

    return node_index;
}

pub fn parseProcedure(self: *Parser) !Ast.NodeIndex {
    std.log.info("begin: {s}", .{@src().fn_name});
    defer std.log.info("end: {s}", .{@src().fn_name});

    var node_index = try self.reserveNode(.procedure);
    errdefer self.unreserveNode(node_index);

    const proto = try self.parseProcedureProto();

    var body = Ast.NodeIndex.nil;

    if (self.peekTokenTag() == .left_brace) {
        body = try self.parseProcedureBody();
    } else {
        _ = try self.expectToken(.semicolon);
    }

    try self.nodeSetData(&node_index, .procedure, .{
        .prototype = proto,
        .body = body,
    });

    std.debug.assert(!proto.isNil());
    std.debug.assert(!node_index.isNil());

    const proc = self.node_heap.getNodePtr(.procedure, node_index.index);

    std.debug.assert(!proc.prototype.isNil());

    return node_index;
}

pub fn parseProcedureBody(self: *Parser) !Ast.NodeIndex {
    std.log.info("begin: {s}", .{@src().fn_name});
    defer std.log.info("end: {s}", .{@src().fn_name});

    var node_index = try self.reserveNode(.procedure_body);
    errdefer self.unreserveNode(node_index);

    _ = try self.expectToken(.left_brace);

    //TODO: use a scratch arena
    var statements: std.ArrayListUnmanaged(Ast.NodeIndex) = .{};
    defer statements.deinit(self.allocator);

    while (self.peekTokenTag().? != .right_brace) {
        const statement = try self.parseStatement();

        try statements.append(self.allocator, statement);
    }

    _ = try self.expectToken(.right_brace);

    try self.nodeSetData(&node_index, .procedure_body, .{
        .statements = try self.node_heap.allocateDupe(self.allocator, Ast.NodeIndex, statements.items),
    });

    return node_index;
}

pub fn parseProcedureProto(self: *Parser) !Ast.NodeIndex {
    std.log.info("begin: {s}", .{@src().fn_name});
    defer std.log.info("end: {s}", .{@src().fn_name});

    var node_index = try self.reserveNode(.procedure_proto);
    errdefer self.unreserveNode(node_index);

    const type_expr = try self.parseTypeExpr();

    const identifier = try self.expectToken(.identifier);

    _ = try self.expectToken(.left_paren);

    const param_list = try self.parseParamList();

    _ = try self.expectToken(.right_paren);

    try self.nodeSetData(&node_index, .procedure_proto, .{
        .return_type = type_expr,
        .name = identifier,
        .param_list = param_list,
    });

    return node_index;
}

pub fn parseParamList(self: *Parser) !Ast.NodeIndex {
    var node = try self.reserveNode(.param_list);
    errdefer self.unreserveNode(node);

    var param_nodes: std.ArrayListUnmanaged(Ast.NodeIndex) = .{};
    defer param_nodes.deinit(self.allocator);

    while (self.peekTokenTag().? != .right_paren) {
        const param = try self.parseParam();

        try param_nodes.append(self.allocator, param);

        _ = self.eatToken(.comma);
    }

    try self.nodeSetData(&node, .param_list, .{
        .params = try self.node_heap.allocateDupe(self.allocator, Ast.NodeIndex, param_nodes.items),
    });

    return node;
}

pub fn parseParam(self: *Parser) !Ast.NodeIndex {
    var node = try self.reserveNode(.param_expr);
    errdefer self.unreserveNode(node);

    const type_expr = try self.parseTypeExpr();
    const param_identifier = try self.expectToken(.identifier);

    try self.nodeSetData(&node, .param_expr, .{
        .type_expr = type_expr,
        .name = param_identifier,
    });

    return node;
}

pub fn parseStatement(self: *Parser) !Ast.NodeIndex {
    switch (self.peekTokenTag().?) {
        .left_brace => {
            //allocate worst case
            var node = try self.reserveNode(.statement_list);
            errdefer self.unreserveNode(node);

            _ = self.nextToken();

            var statements: std.ArrayListUnmanaged(Ast.NodeIndex) = .{};
            defer statements.deinit(self.allocator);

            while (self.peekTokenTag().? != .right_brace) {
                const statement = try self.parseStatement();

                if (statement.isNil()) continue;

                try statements.append(self.allocator, statement);
            }

            _ = try self.expectToken(.right_brace);

            try self.nodeSetData(&node, .statement_list, .{
                .statements = try self.node_heap.allocateDupe(self.allocator, Ast.NodeIndex, statements.items),
            });

            return node;
        },
        .keyword_float,
        .keyword_uint,
        .keyword_int,
        .keyword_void,
        => {
            //allocate worst case
            var node = try self.reserveNode(.statement_var_init);
            errdefer self.unreserveNode(node);

            const type_expr = try self.parseTypeExpr();

            const variable_name = try self.expectToken(.identifier);

            if (self.eatToken(.equals) != null) {
                const expression = try self.parseExpression();

                try self.nodeSetData(&node, .statement_var_init, .{
                    .type_expr = type_expr,
                    .identifier = variable_name,
                    .expression = expression,
                });
            } else {
                try self.nodeSetData(&node, .statement_var_init, .{
                    .type_expr = type_expr,
                    .identifier = variable_name,
                    .expression = Ast.NodeIndex.nil,
                });
            }

            return node;
        },
        .identifier => {
            const variable_name = try self.expectToken(.identifier);

            //allocate worst case
            var node = try self.reserveNode(.statement_assign_equal);
            errdefer self.unreserveNode(node);

            if (self.eatToken(.equals) != null) {
                const expression = try self.parseExpression();
                try self.nodeSetData(&node, .statement_assign_equal, .{
                    .identifier = variable_name,
                    .expression = expression,
                });
            }

            return node;
        },
        .keyword_if => {
            _ = self.nextToken();

            //allocate worst case
            var node = try self.reserveNode(.statement_if);
            errdefer self.unreserveNode(node);

            const cond_expr = try self.parseExpression();

            const taken_statment = try self.parseStatement();

            var not_taken_statment = Ast.NodeIndex.nil;

            const else_keyword = self.eatToken(.keyword_else);

            if (else_keyword) |_| {
                not_taken_statment = try self.parseStatement();
            }

            try self.nodeSetData(&node, .statement_if, .{
                .condition_expression = cond_expr,
                .taken_statement = taken_statment,
                .not_taken_statement = not_taken_statment,
            });

            return node;
        },
        .keyword_return => {
            _ = self.nextToken();

            //allocate worst case
            var node = try self.reserveNode(.statement_return);
            errdefer self.unreserveNode(node);

            var expression: Ast.NodeIndex = Ast.NodeIndex.nil;

            if (self.peekTokenTag().? != .semicolon) {
                expression = try self.parseExpression();
            }

            _ = try self.expectToken(.semicolon);

            try self.nodeSetData(&node, .statement_return, .{
                .expression = expression,
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

pub fn parseExpression(self: *Parser) anyerror!Ast.NodeIndex {
    switch (self.peekTokenTag().?) {
        .identifier,
        .literal_number,
        .keyword_true,
        .keyword_false,
        => {
            switch (self.token_tags[self.token_index + 1]) {
                .plus => {
                    const node = try self.parseBinaryExpression();

                    return node;
                },
                else => {
                    return try self.parseUnaryExpression();
                },
            }
        },
        .left_paren => {
            _ = self.nextToken();

            const node = try self.parseExpression();

            _ = try self.expectToken(.right_paren);

            return node;
        },
        else => return self.unexpectedToken(),
    }

    return 0;
}

pub fn parseUnaryExpression(self: *Parser) anyerror!Ast.NodeIndex {
    var node = try self.reserveNode(.expression_literal_number);
    errdefer self.unreserveNode(node);

    const open_paren = self.eatToken(.left_paren);

    switch (self.peekTokenTag().?) {
        .literal_number,
        .keyword_true,
        .keyword_false,
        => {
            const literal = self.nextToken().?;

            try self.nodeSetData(&node, .expression_literal_number, .{
                .token = literal,
            });
        },
        .identifier => {
            const identifier = try self.expectToken(.identifier);

            try self.nodeSetData(&node, .expression_identifier, .{
                .token = identifier,
            });
        },
        else => return self.unexpectedToken(),
    }

    if (open_paren != null) {
        _ = try self.expectToken(.right_paren);
    }

    return node;
}

pub fn parseBinaryExpression(self: *Parser) anyerror!Ast.NodeIndex {
    var node = try self.reserveNode(.expression_binary_add);
    errdefer self.unreserveNode(node);

    const lhs = try self.parseUnaryExpression();

    switch (self.peekTokenTag().?) {
        .plus => {
            _ = self.nextToken();
        },
        else => return lhs,
    }

    const rhs = try self.parseUnaryExpression();

    std.log.info("node: {}", .{node});
    std.log.info("lhs: {}", .{lhs});
    std.log.info("rhs: {}", .{rhs});

    try self.nodeSetData(&node, .expression_binary_add, .{
        .left = lhs,
        .right = rhs,
    });

    return node;
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

    const context = self.node_context_stack.pop();

    self.token_index = context.saved_token_index;
}

pub fn nodeSetData(
    self: *Parser,
    node: *Ast.NodeIndex,
    comptime Tag: std.meta.Tag(Ast.Node.ExtraData),
    value: std.meta.TagPayload(Ast.Node.ExtraData, Tag),
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
    self.errors.append(self.allocator, .{
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

pub fn peekTokenTag(self: Parser) ?Token.Tag {
    return self.token_tags[self.peekToken() orelse return null];
}

//TODO: support preprocessor directives inside function bodies by modifying this to allow that
pub fn peekToken(self: Parser) ?u32 {
    const result = self.token_index;

    if (result >= self.token_tags.len) {
        return null;
    }

    return result;
}

const std = @import("std");
const Parser = @This();
const Ast = @import("Ast.zig");
const Token = @import("Tokenizer.zig").Token;
