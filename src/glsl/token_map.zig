//!Map like data structures for token strings, handling line continuation

///Comptime token string map containing canonical token strings
///Lookup can be done with non-canonical token strings
pub fn ComptimeCanonicalMap(
    comptime V: type,
    comptime kvs_list: anytype,
) type {
    const empty_list = kvs_list.len == 0;
    const precomputed = blk: {
        @setEvalBranchQuota(1500);
        const KV = struct {
            key: []const u8,
            value: V,
        };
        if (empty_list)
            break :blk .{};
        var sorted_kvs: [kvs_list.len]KV = undefined;
        for (kvs_list, 0..) |kv, i| {
            if (!tokenStringIsCanonical(kv.@"0")) {
                @compileError(std.fmt.comptimePrint("Token string \"{s}\" contains non-canonical characters", .{kv.@"0"}));
            }

            if (V != void) {
                sorted_kvs[i] = .{ .key = kv.@"0", .value = kv.@"1" };
            } else {
                sorted_kvs[i] = .{ .key = kv.@"0", .value = {} };
            }
        }

        const SortContext = struct {
            kvs: []KV,

            pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                return ctx.kvs[a].key.len < ctx.kvs[b].key.len;
            }

            pub fn swap(ctx: @This(), a: usize, b: usize) void {
                return std.mem.swap(KV, &ctx.kvs[a], &ctx.kvs[b]);
            }
        };

        std.mem.sortUnstableContext(0, sorted_kvs.len, SortContext{ .kvs = &sorted_kvs });

        const min_len = sorted_kvs[0].key.len;
        const max_len = sorted_kvs[sorted_kvs.len - 1].key.len;
        var len_indexes: [max_len + 1]usize = undefined;
        var len: usize = 0;
        var i: usize = 0;
        while (len <= max_len) : (len += 1) {
            // find the first keyword len == len
            while (len > sorted_kvs[i].key.len) {
                i += 1;
            }
            len_indexes[len] = i;
        }
        break :blk .{
            .min_len = min_len,
            .max_len = max_len,
            .sorted_kvs = sorted_kvs,
            .len_indexes = len_indexes,
        };
    };

    return struct {
        /// Array of `struct { key: []const u8, value: V }` where `value` is `void{}` if `V` is `void`.
        /// Sorted by `key` length.
        pub const kvs = precomputed.sorted_kvs;

        /// Checks if the map has a value for the key.
        pub fn has(str: []const u8) bool {
            return get(str) != null;
        }

        /// Returns the value for the key if any, else null.
        pub fn get(str: []const u8) ?V {
            if (empty_list)
                return null;

            return precomputed.sorted_kvs[getIndex(str) orelse return null].value;
        }

        pub fn getIndex(str: []const u8) ?usize {
            if (empty_list)
                return null;

            const character_count = tokenStringCharCount(str);

            if (character_count < precomputed.min_len or character_count > precomputed.max_len)
                return null;

            var i = precomputed.len_indexes[character_count];

            while (true) {
                const kv = precomputed.sorted_kvs[i];
                if (kv.key.len != character_count)
                    return null;
                if (tokenStringEqlToCanonical(str, kv.key))
                    return i;
                i += 1;
                if (i >= precomputed.sorted_kvs.len)
                    return null;
            }
        }
    };
}

///Compares two token strings whilst ignoring line continuation digraphs in the non-canonical string
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
