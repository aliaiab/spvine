const std = @import("std");
const Preprocessor = @This();
const Tokenizer = @import("Tokenizer.zig");

pub const GlslVersion = enum
{
    unknown,
    @"450",
};

allocator: std.mem.Allocator,
tokenizer: Tokenizer,
version: GlslVersion,

pub fn init(allocator: std.mem.Allocator, source: []const u8) Preprocessor 
{
    return .{
        .allocator = allocator,
        .tokenizer = .{ .source = source, .index = 0 },
        .version = .unknown,
    };
}

pub fn deinit(self: *Preprocessor) void 
{
    self.* = undefined;
}

pub fn next(self: *Preprocessor) ?Tokenizer.Token
{
    var next_token = self.tokenizer.next() orelse return null;

    switch (next_token.tag) 
    {
        .directive_version => {
            const version_token = self.tokenizer.next() orelse return null;

            std.log.info("FOUND VERSION = {s}", .{ self.tokenizer.source[version_token.start..version_token.end] });        

            return version_token;
        },
        .directive_if => {},
        .directive_endif => {},
        .identifier => {
            //TODO: expand macros
        },
        else => {},
    }

    return next_token;
}

pub fn readAllToEndAlloc(self: *Preprocessor) !std.MultiArrayList(Tokenizer.Token) {
    var tokens = std.MultiArrayList(Tokenizer.Token) {};

    while (self.next()) |token|
    {
        try tokens.append(self.allocator, token);
    }

    return tokens;
}