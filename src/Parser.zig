const std = @import("std");
const Preprocessor = @import("Preprocessor.zig");
const Parser = @This();

const Token = @import("Tokenizer.zig").Token;

allocator: std.mem.Allocator,
preprocessor: Preprocessor,
tokens: std.MultiArrayList(Token),
token_tags: []Token.Tag,
token_starts: []u32,
token_ends: []u32,
token_index: u32,

pub fn init(allocator: std.mem.Allocator, source: []const u8) Parser {
    return .{
        .allocator = allocator,
        .preprocessor = Preprocessor.init(allocator, source),
        .tokens = .{},
        .token_tags = &.{},
        .token_starts = &.{},
        .token_ends = &.{},
        .token_index = 0,
    };
}

pub fn deinit(self: *Parser) void  {
    defer self.* = undefined;
    defer self.preprocessor.deinit();
    defer self.tokens.deinit(self.preprocessor.allocator);
}

pub fn parse(self: *Parser) !void {
    self.tokens = try self.preprocessor.tokenize();
    self.token_tags = self.tokens.items(.tag);
    self.token_starts = self.tokens.items(.start);
    self.token_ends = self.tokens.items(.end);

    var defines = self.preprocessor.defines.iterator();

    while (defines.next()) |define|
    {
        std.log.info("define: {s} = .{s}", .{ define.key_ptr.*, @tagName(self.token_tags[define.value_ptr.start_token]) });
    }

    var state: enum {
        start,
        directive,
    } = .start;

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

                    self.parseProcedure() catch {
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

pub fn parseProcedure(self: *Parser) !void {
    const log = std.log.scoped(.parseProcedure);

    try self.parseType();

    const identifier = try self.expectToken(.identifier);

    log.info("identifier = {s}", .{ self.preprocessor.tokenizer.source[self.tokens.items(.start)[identifier]..self.tokens.items(.end)[identifier]] });

    _ = try self.expectToken(.left_paren);

    //parse arg list

    _ = try self.expectToken(.right_paren);

    _ = try self.expectToken(.left_brace);

    //parse body

    _ = try self.expectToken(.right_brace);
}

pub fn parseType(self: *Parser) !void {
    _ = 
        self.eatToken(.keyword_void) orelse 
        self.eatToken(.keyword_int) orelse
        self.eatToken(.keyword_float) orelse return error.ExpectedToken;
}

pub fn expectToken(self: *Parser, tag: Token.Tag) !u32 {
    return self.eatToken(tag) orelse error.ExpectedToken;
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