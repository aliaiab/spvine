//! Implements the syntactic analysis stage of the frontend

allocator: std.mem.Allocator,
source: []const u8,
token_tags: []const Token.Tag,
token_starts: []const u32,
token_ends: []const u32,
token_index: u32,
nodes: Ast.NodeList,
errors: std.ArrayListUnmanaged(Ast.Error),
extra_data: std.ArrayListUnmanaged(Ast.NodeIndex),

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
        .nodes = .{},
        .errors = .{},
        .extra_data = .{},
    };
}

pub fn deinit(self: *Parser) void {
    defer self.* = undefined;
    defer self.nodes.deinit(self.allocator);
    defer self.errors.deinit(self.allocator);
}

///Root parse node 
pub fn parse(self: *Parser) !void {
    const log = std.log.scoped(.parse);

    _ = try self.reserveNode(.root);

    var state: enum {
        start,
        directive,
    } = .start;

    log.info("begin:", .{});
    defer log.info("end", .{});

    const root_members_start = self.extra_data.items.len; 

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
                .keyword_void,
                .keyword_double,
                .keyword_float,
                .keyword_int,
                .keyword_uint,
                => {
                    std.log.info("Found type expr", .{});

                    const proc = try self.parseProcedure();

                    if (proc != 0) {
                        try self.extra_data.append(self.allocator, proc);
                    }
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

    _ = self.setNode(0, .{
        .tag = .root,
        .data = .{
            .left = @intCast(u32, root_members_start),
            .right = @intCast(u32, self.extra_data.items.len),
        },
        .main_token = 0,
    });
}

pub fn parseProcedure(self: *Parser) !Ast.NodeIndex {
    const log = std.log.scoped(.parseProcedure);

    log.info("begin:", .{});
    defer log.info("end", .{});

    const node_index = try self.reserveNode(.proc_decl);
    errdefer self.unreserveNode(node_index);

    const type_expr = try self.parseTypeExpr();

    _ = type_expr;

    const identifier = self.eatToken(.identifier) orelse return 0;

    log.info("identifier = {s}", .{ self.source[self.token_starts[identifier]..self.token_ends[identifier]] });

    _ = self.expectToken(.left_paren) catch return 0;

    const param_list = self.parseParamList() catch return 0;

    if (param_list == 0) return 0;

    _ = try self.expectToken(.right_paren);

    try self.parseStatement();

    _ = try self.expectToken(.left_brace);

    try self.parseStatement();

    _ = self.expectToken(.right_brace) catch return 0;

    return node_index;
}

pub fn parseParamList(self: *Parser) !Ast.NodeIndex {
    const log = std.log.scoped(.parseParamList);

    log.info("begin:", .{});
    defer log.info("end", .{});

    const node = try self.reserveNode(.param_expr);
    errdefer self.unreserveNode(node);

    while (true) {
        _ = self.parseTypeExpr() catch return 0;

        const param_identifier = self.expectToken(.identifier) catch return 0;

        std.log.info("param_list: param_identifier = {s}", .{ self.source[self.token_starts[param_identifier]..self.token_ends[param_identifier]] });

        if (self.eatToken(.comma) == null)
        {
            break;
        }
    }

    return node;
}

pub fn parseStatementList(self: *Parser) !void { 
    _ = self;    
}

pub fn parseStatement(self: *Parser) !void {
    const log = std.log.scoped(.parseStatement);

    log.info("begin:", .{});
    defer log.info("end", .{});

    switch (self.token_tags[self.token_index]) {
        .keyword_if => {},
        .keyword_else => {},
        .semicolon => {},
        .keyword_float,
        .keyword_int,
        .keyword_void,
        => {
            //int a = 0;
            _ = self.nextToken();
        },
        else => {},
    }
}

pub fn parseExpression(self: *Parser) !Ast.NodeIndex {
    const log = std.log.scoped(.parseExpression);

    log.info("begin:", .{});
    defer log.info("end", .{});

    switch (self.token_tags[self.token_index]) {
        .literal_integer => {},
        .left_paren => {
            self.token_index += 1;

            try self.parseExpression();

            _ = try self.expectToken(.right_paren);
        },
        else => {},
    }
}

pub fn parseTypeExpr(self: *Parser) !Ast.NodeIndex {
    switch (self.token_tags[self.token_index]) {
        .keyword_void,
        .keyword_int,
        .keyword_uint,
        .keyword_float,
        .keyword_double,
        => {
            const node = try self.reserveNode(.type_expr);

            defer self.token_index += 1;

            return self.setNode(node, .{
                .tag = .type_expr,
                .main_token = self.token_index,
                .data = .{ .left = 0, .right = 0, },
            });
        },
        else => {},
    }

    return 0;
}

pub fn addNode(self: *Parser, allocator: std.mem.Allocator, tag: Ast.Node.Tag) !Ast.NodeIndex {
    const index = try self.nodes.addOne(allocator);

    self.nodes.items(.tag)[index] = tag;

    return @intCast(Ast.NodeIndex, index);
}

pub fn setNode(p: *Parser, i: usize, elem: Ast.NodeList.Elem) Ast.NodeIndex {
    p.nodes.set(i, elem);
    return @intCast(Ast.NodeIndex, i);
}

pub fn reserveNode(self: *Parser, tag: Ast.Node.Tag) !Ast.NodeIndex {
    try self.nodes.resize(self.allocator, self.nodes.len + 1);
    self.nodes.items(.tag)[self.nodes.len - 1] = tag;
    return @intCast(Ast.NodeIndex, self.nodes.len - 1);
}

pub fn unreserveNode(self: *Parser, node: Ast.NodeIndex) void 
{
    if (node == self.nodes.len) {
        self.nodes.resize(self.allocator, self.nodes.len - 1) catch unreachable;
    }
    else {
        self.nodes.items(.tag)[node] = .nil;
    }
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

pub fn tokenIndexString(self: Parser, token_index: u32) []const u8 {
    return self.tokenString(self.tokens.get(token_index));
}

pub fn tokenString(self: Parser, token: Token) []const u8 {
    return self.preprocessor.tokenizer.source[token.start..token.end];
} 

pub fn eatToken(self: *Parser, tag: Token.Tag) ?u32 {
    if (self.token_index < self.token_tags.len and self.token_tags[self.token_index] == tag)
    {
        return self.nextToken();
    }
    else 
    {
        return null;
    }
}

pub fn nextToken(self: *Parser) ?u32 {
    const result = self.token_index;
    self.token_index += 1;

    if (result >= self.token_tags.len)
    {
        return null;
    }

    return result;
}

const std = @import("std");
const Preprocessor = @import("Preprocessor.zig");
const Parser = @This();
const Ast = @import("Ast.zig");
const Token = @import("Tokenizer.zig").Token;