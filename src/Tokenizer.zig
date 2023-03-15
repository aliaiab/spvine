const std = @import("std");
const Tokenizer = @This();

source: []const u8,
index: u32,

pub fn next(self: *Tokenizer) ?Token {
    var state: enum {
        start,
        identifier,
        slash,
        single_comment,
        multi_comment,
        literal_integer,
    } = .start;
    
    var token = Token { .start = self.index, .end = self.index, .tag = .end };

    var multi_comment_level: usize = 0;

    while (self.index < self.source.len) : (self.index += 1) {
        const char = self.source[self.index];

        switch (state) {
            .start => switch (char) {
                0 => break,
                ' ', '\t', '\r', '\n' => {
                    token.start = self.index + 1;
                },
                'a'...'z', 'A'...'Z', '_', => {
                    state = .identifier;
                    token.tag = .identifier;
                },
                '0'...'9' => {
                    state = .literal_integer;
                },
                '/' => {
                    state = .slash;
                },
                '{' => {
                    token.tag = .left_brace;
                    self.index += 1;
                    break;
                },
                '}' => {
                    token.tag = .right_brace;
                    self.index += 1;
                    break;
                },
                '(' => {
                    token.tag = .left_paren;
                    self.index += 1;
                    break;
                },
                ')' => {
                    token.tag = .right_paren;
                    self.index += 1;
                    break;
                },
                ';' => {
                    token.tag = .semicolon;
                    self.index += 1;
                    break;
                },
                ',' => {
                    token.tag = .comma;
                    self.index += 1;
                    break;
                },
                '.' => {
                    token.tag = .period;
                    self.index += 1;
                    break;
                },
                '+' => {
                    token.tag = .plus;
                    self.index += 1;
                    break;
                },
                '-' => {
                    token.tag = .minus;
                    self.index += 1;
                    break;
                },
                '=' => {
                    token.tag = .equals;
                    self.index += 1;
                    break;
                },
                else => {},
            },
            .identifier => switch (char) {
                'a'...'z', 'A'...'Z', '_', '0'...'9' => {},
                else => {
                    const string = self.source[token.start..self.index];

                    if (Token.getKeyword(string)) |keyword_tag|
                    {
                        token.tag = keyword_tag;
                    }

                    break;
                },
            },
            .slash => switch (char) {
                '/' => {
                    state = .single_comment;
                },
                '*' => {
                    state = .multi_comment;
                },
                else => {},
            },
            .single_comment => switch (char) {
                '\n' => {
                    state = .start;
                    token.start = self.index + 1;  
                },
                else => {},
            },
            .multi_comment => switch (char) {
                '*' => {
                    if (self.source[self.index + 1] == '/')
                    {
                        self.index += 1;

                        if (multi_comment_level == 0)
                        {
                            state = .start;
                        }
                        else 
                        {
                            multi_comment_level -= 1;
                        }
                    }
                },
                '/' => {
                    if (self.source[self.index + 1] == '*')
                    {
                        self.index += 1;
                        multi_comment_level += 1;
                    }
                },
                else => {},
            },
            .literal_integer => switch (char) {
                '0'...'9' => {},
                else => {
                    token.tag = .literal_integer;

                    break;
                },
            },
        }
    }

    if (token.tag == .end or self.index > self.source.len)
    {
        return null;
    }

    token.end = self.index;

    return token;
}

pub const Token = struct {
    start: u32,
    end: u32,
    tag: Tag,

    pub const Tag = enum {
        end,
        keyword_void,
        keyword_float,
        keyword_if,
        keyword_else,
        keyword_return,
        literal_integer,
        identifier,
        left_brace,
        right_brace,
        left_paren,
        right_paren,
        semicolon,
        comma,
        period,
        plus,
        minus,
        equals,

        pub fn lexeme(tag: Tag) ?[]const u8 {
            return switch (tag) {
                .end,
                .identifier,
                .literal_integer,
                => null,
                .keyword_void => "void",
                .keyword_float => "float",
                .keyword_if => "if",
                .keyword_else => "else",
                .keyword_return => "return",
                .left_brace => "{",
                .right_brace => "}",
                .left_paren => "(",
                .right_paren => ")",
                .semicolon => ";",
                .comma => ",",
                .period => ".",
                .plus => "+",
                .minus => "-",
                .equals => "=",
            };
        }
    };

    pub fn lexeme(token: Token) ?[]const u8
    {
        return token.tag.lexeme();
    }

    pub fn getKeyword(string: []const u8) ?Tag {
        return keywords.get(string);
    }

    const keywords = std.ComptimeStringMap(Tag, .{
        .{ "void", .keyword_void },
        .{ "float", .keyword_float },
        .{ "if", .keyword_if },
        .{ "else", .keyword_else },
        .{ "return", .keyword_return },
    });
};