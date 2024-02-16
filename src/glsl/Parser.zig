//! Implements the syntactic analysis stage of the frontend

allocator: std.mem.Allocator,
source: []const u8,
token_tags: []const Token.Tag,
token_starts: []const u32,
token_ends: []const u32,
token_index: u32,
nodes: Ast.NodeList,
node_context_stack: std.ArrayListUnmanaged(struct {
    saved_token_index: u32,
    saved_error_index: u32,
}),
errors: std.ArrayListUnmanaged(Ast.Error),
extra_data: std.ArrayListUnmanaged(u32),

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
        .nodes = .{},
        .errors = .{},
        .extra_data = .{},
    };
}

pub fn deinit(self: *Parser) void {
    defer self.* = undefined;
    defer self.nodes.deinit(self.allocator);
    defer self.errors.deinit(self.allocator);
    defer self.node_context_stack.deinit(self.allocator);
    defer self.extra_data.deinit(self.allocator);
}

///Root parse node
pub fn parse(self: *Parser) !void {
    _ = try self.reserveNode(.root);

    var state: enum {
        start,
        directive,
    } = .start;

    errdefer {
        _ = self.setNode(0, .{
            .tag = .root,
            .data = .{
                .left = 0,
                .right = 0,
            },
            .main_token = 0,
        });
    }

    var root_nodes: std.ArrayListUnmanaged(Ast.NodeIndex) = .{};
    defer root_nodes.deinit(self.allocator);

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

    const root_members_start = self.extra_data.items.len;

    try self.extra_data.appendSlice(self.allocator, root_nodes.items);

    _ = self.setNode(0, .{
        .tag = .root,
        .data = .{
            .left = @as(u32, @intCast(root_members_start)),
            .right = @as(u32, @intCast(self.extra_data.items.len)),
        },
        .main_token = 0,
    });
}

pub fn parseStruct(self: *Parser) !Ast.NodeIndex {
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

pub fn parseProcedureProto(self: *Parser) !Ast.NodeIndex {
    const node_index = try self.reserveNode(.procedure_proto);
    errdefer self.unreserveNode(node_index);

    const type_expr = try self.parseTypeExpr();

    const identifier = try self.expectToken(.identifier);

    _ = try self.expectToken(.left_paren);

    const param_list = try self.parseParamList();

    _ = try self.expectToken(.right_paren);

    try self.nodeSetData(node_index, .procedure_proto, .{
        .return_type = type_expr,
        .name = identifier,
        .param_list = param_list,
    });

    return node_index;
}

pub fn parseProcedureBody(self: *Parser) !Ast.NodeIndex {
    const node_index = try self.reserveNode(.procedure_body);
    errdefer self.unreserveNode(node_index);

    _ = try self.expectToken(.left_brace);

    var statements: std.ArrayListUnmanaged(Ast.NodeIndex) = .{};
    defer statements.deinit(self.allocator);

    while (self.peekTokenTag().? != .right_brace) {
        const statement = try self.parseStatement();

        try statements.append(self.allocator, statement);
    }

    _ = try self.expectToken(.right_brace);

    try self.nodeSetData(node_index, .procedure_body, .{
        .statements = statements.items,
    });

    return node_index;
}

pub fn parseProcedure(self: *Parser) !Ast.NodeIndex {
    const node_index = try self.reserveNode(.procedure);
    errdefer self.unreserveNode(node_index);

    const proto = try self.parseProcedureProto();
    const body = try self.parseProcedureBody();

    try self.nodeSetData(node_index, .procedure, .{
        .prototype = proto,
        .body = body,
    });

    return node_index;
}

pub fn parseParamList(self: *Parser) !Ast.NodeIndex {
    const node = try self.reserveNode(.param_list);
    errdefer self.unreserveNode(node);

    var param_nodes: std.ArrayListUnmanaged(Ast.NodeIndex) = .{};
    defer param_nodes.deinit(self.allocator);

    while (self.peekTokenTag().? != .right_paren) {
        const param = try self.parseParam();

        try param_nodes.append(self.allocator, param);

        _ = self.eatToken(.comma);
    }

    try self.nodeSetData(node, .param_list, .{
        .params = param_nodes.items,
    });

    return node;
}

pub fn parseParam(self: *Parser) !Ast.NodeIndex {
    const node = try self.reserveNode(.param_expr);
    errdefer self.unreserveNode(node);

    const type_expr = try self.parseTypeExpr();
    const param_identifier = try self.expectToken(.identifier);

    try self.nodeSetData(node, .param_expr, .{
        .type_expr = type_expr,
        .name = param_identifier,
    });

    return node;
}

pub fn parseStatementList(self: *Parser) !void {
    _ = self;
}

pub fn parseStatement(self: *Parser) !Ast.NodeIndex {
    const node = try self.reserveNode(.statement);
    errdefer self.unreserveNode(node);

    switch (self.peekTokenTag().?) {
        .keyword_float,
        .keyword_uint,
        .keyword_int,
        .keyword_void,
        => {
            _ = self.nextToken();

            _ = try self.expectToken(.identifier);

            if (self.eatToken(.equals) != null) {
                _ = try self.parseExpression();
            }
        },
        .keyword_return => {
            _ = self.nextToken();

            _ = try self.parseExpression();
        },
        else => return error.ExpectedToken,
    }

    _ = try self.expectToken(.semicolon);

    return node;
}

pub fn parseExpression(self: *Parser) anyerror!Ast.NodeIndex {
    switch (self.peekTokenTag().?) {
        .identifier,
        .literal_number,
        => {
            switch (self.token_tags[self.token_index + 1]) {
                .plus => {
                    _ = try self.parseBinaryExpression();
                },
                else => {
                    _ = try self.parseUnaryExpression();
                },
            }
        },
        .left_paren => {
            _ = self.nextToken();

            _ = try self.parseExpression();

            _ = try self.expectToken(.right_paren);
        },
        else => unreachable,
    }

    return 0;
}

pub fn parseUnaryExpression(self: *Parser) anyerror!Ast.NodeIndex {
    const node = try self.reserveNode(.nil);
    errdefer self.unreserveNode(node);

    const open_paren = self.eatToken(.left_paren);

    switch (self.peekTokenTag().?) {
        .literal_number => {
            _ = self.nextToken();
        },
        .identifier => {
            _ = self.nextToken();
        },
        else => return error.ExpectedToken,
    }

    if (open_paren != null) {
        _ = try self.expectToken(.right_paren);
    }

    return node;
}

pub fn parseBinaryExpression(self: *Parser) anyerror!Ast.NodeIndex {
    const node = try self.reserveNode(.nil);
    errdefer self.unreserveNode(node);

    const lhs = try self.parseUnaryExpression();

    switch (self.peekTokenTag().?) {
        .plus => {
            _ = self.nextToken();
        },
        else => return lhs,
    }

    const rhs = try self.parseUnaryExpression();
    _ = rhs; // autofix

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
            const node = try self.reserveNode(.type_expr);
            errdefer self.unreserveNode(node);

            defer self.token_index += 1;

            return self.setNode(node, .{
                .tag = .type_expr,
                .main_token = self.token_index,
                .data = .{
                    .left = self.token_index,
                    .right = 0,
                },
            });
        },
        else => {},
    }

    return 0;
}

pub fn addNode(self: *Parser, allocator: std.mem.Allocator, tag: Ast.Node.Tag) !Ast.NodeIndex {
    const index = try self.nodes.addOne(allocator);

    self.nodes.items(.tag)[index] = tag;

    return @as(Ast.NodeIndex, @intCast(index));
}

pub fn setNode(p: *Parser, i: usize, elem: Ast.Node) Ast.NodeIndex {
    p.nodes.set(i, elem);
    return @as(Ast.NodeIndex, @intCast(i));
}

pub fn reserveNode(self: *Parser, tag: Ast.Node.Tag) !Ast.NodeIndex {
    try self.nodes.resize(self.allocator, self.nodes.len + 1);
    self.nodes.items(.tag)[self.nodes.len - 1] = tag;

    try self.node_context_stack.append(self.allocator, .{
        .saved_token_index = self.token_index,
        .saved_error_index = @intCast(self.errors.items.len),
    });

    return @as(Ast.NodeIndex, @intCast(self.nodes.len - 1));
}

pub fn unreserveNode(self: *Parser, node: Ast.NodeIndex) void {
    if (node == self.nodes.len) {
        self.nodes.resize(self.allocator, self.nodes.len - 1) catch unreachable;
    } else {
        self.nodes.items(.tag)[node] = .nil;
    }

    const context = self.node_context_stack.pop();

    self.token_index = context.saved_token_index;
    // self.errors.items.len = context.saved_error_index;
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
        .token = self.token_index - 1,
        .data = .{
            .expected_token = tag,
        },
    }) catch unreachable;

    return self.eatToken(tag) orelse error.ExpectedToken;
}

pub fn eatToken(self: *Parser, tag: Token.Tag) ?u32 {
    if (self.token_index < self.token_tags.len and self.token_tags[self.token_index] == tag) {
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

pub fn peekToken(self: Parser) ?u32 {
    const result = self.token_index;

    if (result >= self.token_tags.len) {
        return null;
    }

    return result;
}

pub fn peekTokenTag(self: Parser) ?Token.Tag {
    return self.token_tags[self.peekToken() orelse return null];
}

pub fn nodeSetData(
    self: *Parser,
    node: Ast.NodeIndex,
    comptime Tag: std.meta.Tag(Ast.Node.ExtraData),
    value: std.meta.TagPayload(Ast.Node.ExtraData, Tag),
) !void {
    const extra_data_start = self.extra_data.items.len;

    if (std.meta.fields(@TypeOf(value)).len <= 2) {
        inline for (std.meta.fields(@TypeOf(value)), 0..) |field, field_index| {
            switch (field.type) {
                u32 => {
                    switch (std.meta.fields(@TypeOf(value)).len) {
                        0 => {},
                        1 => {
                            self.nodes.items(.data)[node].left = @field(value, field.name);
                        },
                        2 => {
                            if (field_index == 0) {
                                self.nodes.items(.data)[node].left = @field(value, field.name);
                            } else {
                                self.nodes.items(.data)[node].right = @field(value, field.name);
                            }
                        },
                        else => @compileError("."),
                    }
                },
                []const u32 => {
                    if (std.meta.fields(@TypeOf(value)).len > 1) {
                        @compileError("Multiple slices not yet supported");
                    }

                    const data_start = self.extra_data.items.len;

                    try self.extra_data.appendSlice(self.allocator, @field(value, field.name));

                    const extra_data_end = self.extra_data.items.len;

                    self.nodes.items(.data)[node] = .{
                        .left = @intCast(data_start),
                        .right = @intCast(extra_data_end),
                    };
                },
                else => @compileError("Type not supported"),
            }
        }

        return;
    }

    try self.extra_data.ensureTotalCapacity(self.allocator, self.extra_data.items.len + std.meta.fields(@TypeOf(value)).len);

    inline for (std.meta.fields(@TypeOf(value))) |field| {
        switch (field.type) {
            u32 => {
                try self.extra_data.append(self.allocator, @field(value, field.name));
            },
            []const u32 => @compileError("Not yet supported"),
            else => @compileError("Type not supported"),
        }
    }

    const extra_data_end = self.extra_data.items.len;

    self.nodes.items(.data)[node] = .{
        .left = @intCast(extra_data_start),
        .right = @intCast(extra_data_end),
    };
}

const std = @import("std");
const Parser = @This();
const Ast = @import("Ast.zig");
const Token = @import("Tokenizer.zig").Token;
