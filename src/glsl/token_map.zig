//!Map like data structures for token strings, handling line continuation

///Comptime token string map containing canonical token strings
///Lookup can be done with non-canonical token strings
pub fn StaticCanonicalMap(
    comptime V: type,
) type {
    return std.static_string_map.StaticStringMapWithEql(V, tokenStringEql);
}

pub fn Map(comptime V: type) type {
    return std.HashMapUnmanaged([]const u8, V, TokenStringContext, 80);
}

pub const TokenStringContext = struct {
    pub fn eql(_: @This(), a: []const u8, b: []const u8) bool {
        return tokenStringEql(a, b);
    }

    pub fn hash(_: @This(), a: []const u8) u64 {
        return tokenStringHash(a);
    }
};

///Compares two token strings whilst ignoring line continuation digraphs in the non-canonical string
///The canonical string must not have line continuation characters, eg the literal "void" not "voi\\nd"
pub fn tokenStringEqlToCanonical(a: []const u8, canonical: []const u8) bool {
    if (a.ptr == canonical.ptr) return true;

    var a_index: usize = 0;
    var b_index: usize = 0;

    while (a_index < a.len and b_index < canonical.len) {
        switch (a[a_index]) {
            '\\', '\n', '\r' => {
                a_index += 1;
                continue;
            },
            else => {},
        }

        const a_elem = a[a_index];
        const b_elem = canonical[b_index];

        if (a_elem != b_elem) return false;

        a_index += 1;
        b_index += 1;
    }

    return true;
}

///Compares two token strings whilst ignoring line continuation digraphs
pub fn tokenStringEql(a: []const u8, b: []const u8) bool {
    if (a.ptr == b.ptr) return true;

    var a_index: usize = 0;
    var b_index: usize = 0;

    while (a_index < a.len and b_index < b.len) {
        switch (a[a_index]) {
            '\\', '\n', '\r' => {
                a_index += 1;
                continue;
            },
            else => {},
        }

        switch (b[b_index]) {
            '\\', '\n', '\r' => {
                b_index += 1;
                continue;
            },
            else => {},
        }

        const a_elem = a[a_index];
        const b_elem = b[b_index];

        if (a_elem != b_elem) return false;

        a_index += 1;
        b_index += 1;
    }

    return true;
}

///Returns the number of characters that contribute to the identity of the token string
pub fn tokenStringCharCount(string: []const u8) usize {
    var count: usize = 0;

    for (string) |char| {
        if (char == '\n' or char == '\\' or char == '\r') {
            continue;
        }

        count += 1;
    }

    return count;
}

///Returns true if the string contains no exceptional characters (eg new lines and backslashes)
pub fn tokenStringIsCanonical(string: []const u8) bool {
    return tokenStringCharCount(string) == string.len;
}

///TODO: use a better hash function?
///Computes the hash of the string as if the string contained no line continuation digraphs, if it has any.
pub fn tokenStringHash(str: []const u8) u64 {
    var hash: u32 = 0;

    const b = 255;
    const m = 1000000009;

    for (str) |char| {
        if (char == '\n' or char == '\\' or char == '\n') continue;

        hash = (hash *% b +% char) % m;
    }

    return hash;
}

test "Token string equality" {
    const expect = std.testing.expect;

    try expect(tokenStringEqlToCanonical("void", "void"));
    try expect(tokenStringEqlToCanonical("v\\\noid", "void"));
    try expect(tokenStringEqlToCanonical("v\\\n oid", "void") == false);

    try expect(tokenStringEql("void", "void"));
    try expect(tokenStringEql("v\\\noid", "void"));
    try expect(tokenStringEql("v\\\n oid", "void") == false);

    try expect(tokenStringEql("void", "void"));
    try expect(tokenStringEql("void", "v\\\noid"));
    try expect(tokenStringEql("v\\\noid", "void"));
    try expect(tokenStringEql("v\\\noid", "v\\\noid"));
    try expect(tokenStringEql("v\\\n oid", "void") == false);
}

test "Token string hash" {
    const expect = std.testing.expect;

    try expect(tokenStringHash("void") == tokenStringHash("void"));
    try expect(tokenStringHash("v\\\noid") == tokenStringHash("void"));
    try expect(tokenStringHash("v\\\n oid") != tokenStringHash("void"));

    try expect(tokenStringHash("f32") == tokenStringHash("f32"));
    try expect(tokenStringHash("f32") != tokenStringHash("f64"));
}

const std = @import("std");
