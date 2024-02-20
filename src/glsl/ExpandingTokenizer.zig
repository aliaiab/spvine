//! Implements the glsl preprocessor directives, which expands macros into larger token streams

allocator: std.mem.Allocator,
defines: DefineMap,
tokenizer: Tokenizer,

pub const Define = struct {
    start_token: u32,
};

pub fn init(allocator: std.mem.Allocator, source: []const u8) ExpandingTokenizer {
    return .{
        .allocator = allocator,
        .defines = .{},
        .tokenizer = .{ .source = source },
    };
}

pub fn deinit(self: *ExpandingTokenizer) void {
    self.defines.deinit(self.allocator);
    self.* = undefined;
}

///TODO: remove the need to store tokens entirely, by calling the tokenizer during parsing
pub fn tokenize(
    self: *ExpandingTokenizer,
    tokens: *Ast.TokenList,
    errors: *std.ArrayListUnmanaged(Ast.Error),
) !void {
    var if_condition: bool = true;
    var if_condition_level: u32 = 0;

    var current_if_level: u32 = 0;

    while (self.tokenizer.next()) |token| {
        switch (token.tag) {
            .directive_version => {
                if (!if_condition) continue;

                const string = "__VERSION__";

                const define = self.defines.getOrPut(self.allocator, string) catch unreachable;

                try tokens.append(self.allocator, token);

                const start_token = @as(u32, @intCast(tokens.len));

                define.value_ptr.start_token = start_token;

                try tokens.append(self.allocator, self.tokenizer.next() orelse break);
            },
            .directive_if => {
                current_if_level += 1;

                if (!if_condition) continue;

                const identifier_token = self.tokenizer.next() orelse break;

                const string = self.tokenizer.source[identifier_token.start..identifier_token.end];

                const define = self.defines.get(string) orelse return error.UndefinedMacro;

                const value_token = tokens.get(define.start_token);

                const value = try std.fmt.parseUnsigned(u64, self.tokenizer.source[value_token.start..value_token.end], 10);

                if_condition = value == 1;
                if_condition_level += 1;
            },
            .directive_ifdef => {
                current_if_level += 1;

                if (!if_condition) continue;

                const identifier_token = self.tokenizer.next() orelse break;

                const string = self.tokenizer.source[identifier_token.start..identifier_token.end];

                if_condition = self.defines.contains(string);
                if_condition_level += 1;
            },
            .directive_ifndef => {
                current_if_level += 1;

                if (!if_condition) continue;

                const identifier_token = self.tokenizer.next() orelse break;

                const string = self.tokenizer.source[identifier_token.start..identifier_token.end];

                if_condition = !self.defines.contains(string);
                if_condition_level += 1;
            },
            .directive_endif => {
                if_condition = true;
                if_condition_level -= 1;
                current_if_level -= 1;
            },
            .directive_define => {
                if (!if_condition) continue;

                const identifier_token = self.tokenizer.next() orelse break;

                const string = self.tokenizer.source[identifier_token.start..identifier_token.end];

                const define = self.defines.getOrPut(self.allocator, string) catch unreachable;

                try tokens.append(self.allocator, token);
                try tokens.append(self.allocator, identifier_token);

                const start_token = @as(u32, @intCast(tokens.len));

                define.value_ptr.start_token = start_token;

                try tokens.append(self.allocator, self.tokenizer.next() orelse break);
            },
            .directive_undef => {
                if (!if_condition) continue;

                const identifier_token = self.tokenizer.next() orelse break;

                _ = self.defines.remove(self.tokenizer.source[identifier_token.start..identifier_token.end]);
            },
            .directive_error => {
                if (!if_condition) continue;

                try errors.append(self.allocator, .{
                    .tag = .directive_error,
                    .token = @as(u32, @intCast(tokens.len)),
                });

                try tokens.append(self.allocator, token);
            },
            .directive_end => {
                if (!if_condition) continue;

                try tokens.append(self.allocator, token);
            },
            .identifier => {
                if (!if_condition) continue;

                if (try self.tryExpandMacro(tokens, token)) {} else {
                    try tokens.append(self.allocator, token);
                }
            },
            .directive_line, .directive_include => {
                try errors.append(self.allocator, .{
                    .tag = .unsupported_directive,
                    .token = @intCast(tokens.len),
                });

                try tokens.append(self.allocator, token);
            },
            else => {
                if (!if_condition) continue;

                try tokens.append(self.allocator, token);
            },
        }
    }
}

fn tryExpandMacro(self: *ExpandingTokenizer, tokens: *std.MultiArrayList(Tokenizer.Token), identifier_token: Tokenizer.Token) !bool {
    const string = self.tokenizer.source[identifier_token.start..identifier_token.end];

    const define: Define = self.defines.get(string) orelse return false;

    var token_index: u32 = define.start_token;

    while (token_index < tokens.len) {
        const token = tokens.get(token_index);

        if (token.tag == .directive_end and token_index == define.start_token) {
            try tokens.append(self.allocator, identifier_token);

            return true;
        }

        switch (token.tag) {
            .identifier => {
                if (!try self.tryExpandMacro(tokens, token)) {
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

pub const DefineMap = std.StringHashMapUnmanaged(Define);

const std = @import("std");
const ExpandingTokenizer = @This();
const Tokenizer = @import("Tokenizer.zig");
const Ast = @import("Ast.zig");
