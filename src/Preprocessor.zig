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
defines: std.StringArrayHashMapUnmanaged(Define),

pub const Define = struct 
{
    start_token: Tokenizer.Token,
};

pub fn init(allocator: std.mem.Allocator, source: []const u8) Preprocessor 
{
    return .{
        .allocator = allocator,
        .tokenizer = .{ .source = source, .index = 0 },
        .version = .unknown,
        .defines = .{},
    };
}

pub fn deinit(self: *Preprocessor) void 
{
    defer self.* = undefined;
    defer self.defines.deinit(self.allocator);
}

pub fn tokenize(self: *Preprocessor) !std.MultiArrayList(Tokenizer.Token) {
    var tokens = std.MultiArrayList(Tokenizer.Token) {};

    while (self.tokenizer.next()) |token|
    {
        switch (token.tag) 
        {
            .directive_version => {
                const version_token = self.tokenizer.next() orelse break;

                std.log.info("FOUND VERSION = {s}", .{ self.tokenizer.source[version_token.start..version_token.end] });        
            },
            .directive_if => {
                const identifier_token = self.tokenizer.next() orelse break;

                const string = self.tokenizer.source[identifier_token.start..identifier_token.end];

                const define = self.defines.get(string) orelse return error.UndefinedMacro;

                const value_token = define.start_token;

                _ = value_token;
            },
            .directive_endif => {},
            .directive_define => {
                const identifier_token = self.tokenizer.next() orelse break;

                const string = self.tokenizer.source[identifier_token.start..identifier_token.end];

                const define = self.defines.getOrPut(self.allocator, string) catch unreachable;

                define.value_ptr.start_token = self.tokenizer.next() orelse break;
            },
            .identifier => {
                //TODO: expand macros

                // try expandMacro(token);
            },
            else => {},
        }

        try tokens.append(self.allocator, token);
    }

    return tokens;
}

fn expandMacro(self: *Preprocessor, tokens: std.MultiArrayList(Tokenizer.Token), identifier_token: Tokenizer.Token) !void 
{
    const string = self.tokenizer.source[identifier_token.start..identifier_token.end];

    const define: Define = self.defines.get(self.allocator, string) catch unreachable;

    const start_token = define.start_token;

    _ = start_token;
    _ = tokens;
}