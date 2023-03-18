//! Implements the glsl preprocessor, which expands macros into larger token streams 

allocator: std.mem.Allocator,
tokenizer: Tokenizer,
defines: std.StringArrayHashMapUnmanaged(Define),

pub const Define = struct {
    start_token: u32,
};

pub fn init(allocator: std.mem.Allocator, source: []const u8) Preprocessor {
    return .{
        .allocator = allocator,
        .tokenizer = .{ .source = source, .index = 0 },
        .defines = .{},
    };
}

pub fn deinit(self: *Preprocessor) void {
    defer self.* = undefined;
    defer self.defines.deinit(self.allocator);
}

pub fn tokenize(self: *Preprocessor, tokens: *Ast.TokenList) !void {
    while (self.tokenizer.next()) |token| {
        switch (token.tag) {
            .directive_version => {},
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

                try tokens.append(self.allocator, token);
                try tokens.append(self.allocator, identifier_token);

                const start_token = @intCast(u32, tokens.len);

                define.value_ptr.start_token = start_token;

                try tokens.append(self.allocator, self.tokenizer.next() orelse break);
            },
            .directive_end => {
                try tokens.append(self.allocator, token);
            },
            .identifier => {
                if (try self.tryExpandMacro(tokens, token)) {

                } else {
                    try tokens.append(self.allocator, token);
                }
            },
            else => {
                try tokens.append(self.allocator, token);
            },
        }
    }
}

fn tryExpandMacro(self: *Preprocessor, tokens: *std.MultiArrayList(Tokenizer.Token), identifier_token: Tokenizer.Token) !bool {
    const string = self.tokenizer.source[identifier_token.start..identifier_token.end];

    const define: Define = self.defines.get(string) orelse return false;

    var token_index: u32 = define.start_token;

    while (token_index < tokens.len) {
        const token = tokens.get(token_index);

        switch (token.tag)
        {
            .identifier => {
                if (!try self.tryExpandMacro(tokens, token))
                {
                    try tokens.append(self.allocator, token);
                }
                token_index += 1;
            },
            .directive_end => break,
            else => {
                try tokens.append(self.allocator, token);
                token_index += 1;
            },
        }
    }

    return true;
}

const std = @import("std");
const Preprocessor = @This();
const Tokenizer = @import("Tokenizer.zig");
const Ast = @import("Ast.zig");