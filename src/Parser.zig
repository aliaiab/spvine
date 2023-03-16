const std = @import("std");
const Preprocessor = @import("Preprocessor.zig");
const Parser = @This();

const Token = @import("Tokenizer.zig").Token;

allocator: std.mem.Allocator,
preprocessor: Preprocessor,
tokens: std.MultiArrayList(Token),

pub fn init(allocator: std.mem.Allocator, source: []const u8) Parser
{
    return .{
        .allocator = allocator,
        .preprocessor = Preprocessor.init(allocator, source),
        .tokens = .{},
    };
}

pub fn deinit(self: *Parser) void 
{
    defer self.* = undefined;
    defer self.preprocessor.deinit();
    defer self.tokens.deinit(self.preprocessor.allocator);
}

pub fn parse(self: *Parser) !void
{
    self.tokens = try self.preprocessor.readAllToEndAlloc();

    for (self.tokens.items(.tag), self.tokens.items(.start), self.tokens.items(.end)) |tag, start, end|
    {
        std.log.info("{s} - {}:{} ({s})", .{ 
            Token.lexeme(.{ .tag = tag, .start = start, .end = end }) orelse @tagName(tag), 
            start, end, 
            self.preprocessor.tokenizer.source[start..end] 
        });
    }
}