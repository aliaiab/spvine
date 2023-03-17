//! Implements the syntactic analysis stage of the frontend

allocator: std.mem.Allocator,
preprocessor: Preprocessor,
tokens: std.MultiArrayList(Token),
token_tags: []Token.Tag,
token_starts: []u32,
token_ends: []u32,
token_index: u32,
errors: std.ArrayListUnmanaged(Error),
ast: Ast = .{},

pub const Error = struct {
    tag: Tag,
    token: Ast.TokenIndex,
    data: union {
        none: void,
        expected_token: Token.Tag,
    } = .{ .none = {} },

    pub const Tag = enum(u8) {
        expected_token,
    };
};

pub fn init(allocator: std.mem.Allocator, source: []const u8) Parser {
    return .{
        .allocator = allocator,
        .preprocessor = Preprocessor.init(allocator, source),
        .tokens = .{},
        .token_tags = &.{},
        .token_starts = &.{},
        .token_ends = &.{},
        .token_index = 0,
        .errors = .{},
    };
}

pub fn deinit(self: *Parser) void  {
    defer self.* = undefined;
    defer self.preprocessor.deinit();
    defer self.tokens.deinit(self.preprocessor.allocator);
    defer self.ast.deinit(self.allocator);
    defer self.errors.deinit(self.allocator);
}

///Root parse node 
pub fn parse(self: *Parser) !void {
    const log = std.log.scoped(.parse);

    self.tokens = try self.preprocessor.tokenize();
    self.token_tags = self.tokens.items(.tag);
    self.token_starts = self.tokens.items(.start);
    self.token_ends = self.tokens.items(.end);

    var defines = self.preprocessor.defines.iterator();

    while (defines.next()) |define|
    {
        log.info("define: {s} = .{s}", .{ define.key_ptr.*, @tagName(self.token_tags[define.value_ptr.start_token]) });
    }

    var state: enum {
        start,
        directive,
    } = .start;

    log.info("begin:", .{});
    defer log.info("end", .{});

    while (self.token_index < self.tokens.len) {
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
                .keyword_float,
                .keyword_int,
                => {
                    std.log.info("Found type expr", .{});

                    _ = self.parseProcedure() catch {
                        _ = self.nextToken();
                    };
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
}

pub fn parseProcedure(self: *Parser) !Ast.NodeIndex {
    const log = std.log.scoped(.parseProcedure);

    log.info("begin:", .{});
    defer log.info("end", .{});

    const node_index = try self.reserveNode(.proc_decl);
    errdefer self.unreserveNode(node_index);

    try self.parseType();

    const identifier = try self.expectToken(.identifier);

    log.info("identifier = {s}", .{ self.preprocessor.tokenizer.source[self.tokens.items(.start)[identifier]..self.tokens.items(.end)[identifier]] });

    _ = try self.expectToken(.left_paren);

    self.parseParamList() catch {};

    _ = try self.expectToken(.right_paren);

    _ = try self.expectToken(.left_brace);

    try self.parseStatement();

    _ = try self.expectToken(.right_brace);

    return node_index;
}

pub fn parseParamList(self: *Parser) !void {
    const log = std.log.scoped(.parseParamList);

    log.info("begin:", .{});
    defer log.info("end", .{});

    while (true) {
        _ = try self.parseType();

        const param_identifier = try self.expectToken(.identifier);

        std.log.info("param_list: param_identifier = {s}", .{ self.preprocessor.tokenizer.source[self.token_starts[param_identifier]..self.token_ends[param_identifier]] });

        if (self.eatToken(.comma) == null)
        {
            break;
        }
    }
}

pub fn parseStatement(self: *Parser) !void 
{
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

pub fn parseExpression(self: *Parser) !Ast.NodeIndex
{
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

pub fn parseType(self: *Parser) !void {
    _ = 
        self.eatToken(.keyword_void) orelse 
        self.eatToken(.keyword_int) orelse
        self.eatToken(.keyword_float) orelse return error.ExpectedToken;
}

pub fn reserveNode(self: *Parser, tag: Ast.NodeTag) !Ast.NodeIndex {
    return try self.ast.addNode(self.allocator, tag);
}

pub fn unreserveNode(self: *Parser, node: Ast.NodeIndex) void 
{
    if (node == self.ast.nodes.len) {
        self.ast.nodes.resize(self.allocator, self.ast.nodes.len - 1) catch unreachable;
    }
    else {
        self.ast.nodes.items(.tag)[node] = .nil;
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
    if (self.token_index < self.tokens.len and self.tokens.items(.tag)[self.token_index] == tag)
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

    if (result >= self.tokens.len)
    {
        return null;
    }

    const token = self.tokens.get(result);

    std.log.info("{s} - {}:{} ({s})", .{ 
        token.lexeme() orelse @tagName(token.tag), 
        token.start, token.end, 
        self.preprocessor.tokenizer.source[token.start..token.end] 
    });

    return result;
}

const std = @import("std");
const Preprocessor = @import("Preprocessor.zig");
const Parser = @This();
const Ast = @import("Ast.zig");
const Token = @import("Tokenizer.zig").Token;