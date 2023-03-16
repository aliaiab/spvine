const std = @import("std");
const Preprocessor = @This();
const Tokenizer = @import("Tokenizer.zig");

pub const GlslVersion = enum
{
    unknown,
    @"450",
};

tokenizer: Tokenizer,
version: GlslVersion,

pub fn init(source: []const u8) Preprocessor 
{
    return .{
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
        },
        else => {},
    }

    return next_token;
}
