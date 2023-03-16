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
        directive_begin,
        directive_body,
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
                '#' => {
                    state = .directive_begin;
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
            .directive_begin => switch (char) {
                'a'...'z', 'A'...'Z', '_', => {

                },
                else => {
                    state = .directive_body;
                },
            },
            .directive_body => switch (char) {
                '\n' => {
                    token.tag = .directive_end;
                    self.index += 1;
                    break;
                },
                else => {},
            },
            .identifier => switch (char) {
                'a'...'z', 'A'...'Z', '_', '0'...'9' => {},
                else => {
                    const string = self.source[token.start..self.index];

                    if (Token.getDirective(string)) |directive_tag|
                    {
                        token.tag = directive_tag;
                    }

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
        directive_define,
        directive_undef,
        directive_if,
        directive_ifdef,
        directive_ifndef,
        directive_else,
        directive_elif,
        directive_endif,
        directive_error,
        directive_pragma,
        directive_extension,
        directive_version,
        directive_line,
        directive_end,
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
                .directive_end,
                => null,
                .directive_define => "#define",
                .directive_undef => "#undef",
                .directive_if => "#if",
                .directive_ifdef => "#ifdef",
                .directive_ifndef => "#ifndef",
                .directive_else => "#else",
                .directive_elif => "#elif",
                .directive_endif => "#endif",
                .directive_error => "#error",
                .directive_pragma => "#pragma",
                .directive_extension => "#extension",
                .directive_version => "#version",
                .directive_line => "#line",
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

    pub fn getDirective(string: []const u8) ?Tag {
        return directives.get(string);
    }

    const keywords = std.ComptimeStringMap(Tag, .{
        .{ "void", .keyword_void },
        .{ "float", .keyword_float },
        .{ "if", .keyword_if },
        .{ "else", .keyword_else },
        .{ "return", .keyword_return },
    });

    const directives = std.ComptimeStringMap(Tag, .{
        .{ "#define", .directive_define },
        .{ "#undef", .directive_undef },
        .{ "#if", .directive_if },
        .{ "#ifdef", .directive_ifdef },
        .{ "#ifndef", .directive_ifdef },
        .{ "#else", .directive_else },
        .{ "#elif", .directive_elif },
        .{ "#endif", .directive_elif },
        .{ "#error", .directive_error },
        .{ "#pragma", .directive_pragma },
        .{ "#extension", .directive_pragma },
        .{ "#version", .directive_version },
        .{ "#line", .directive_line },
    });
};