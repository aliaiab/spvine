const Tokenizer = @This();

source: []const u8,
index: u32,

pub fn next(self: *Tokenizer) ?Token {
    var state: enum {
        start,
        identifier,
    } = .start;
    
    var token = Token { .start = self.index, .end = self.index, .tag = .end };

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
                else => {},
            },
            .identifier => switch (char) {
                'a'...'z', 'A'...'Z', '_', '0'...'9' => {},
                else => {
                    const string = self.source[token.start..self.index];

                    _ = string;
                    //TODO: handle keywords

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

pub const Token = struct  {
    start: u32,
    end: u32,
    tag: Tag,

    pub const Tag = enum {
        end,
        keyword_void,
        identifier,
        left_brace,
        right_brace,
        left_paren,
        right_paren,
    };
};