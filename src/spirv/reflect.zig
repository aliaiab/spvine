pub const Result = struct {
    entry_point: []const u8,
    source_language: spirv.SourceLanguage,
    execution_model: spirv.ExecutionModel,
    addressing_mode: spirv.AddressingMode,
    memory_model: spirv.MemoryModel,
    execution_mode: spirv.ExecutionMode,

    local_size_x: u32 = 0,
    local_size_y: u32 = 0,
    local_size_z: u32 = 0,
    resources: []Resource,

    const Resource = struct {
        descriptor_type: spirv.Op,
        descriptor_count: u32,
        binding: u32,
    };
};

pub fn parse(allocator: std.mem.Allocator, module: []align(4) const u8) !void {
    var iterator = Iterator.initFromSlice(module);

    var result = Result{
        .resources = &.{},
        .entry_point = "",
        .source_language = undefined,
        .execution_model = undefined,
        .addressing_mode = undefined,
        .memory_model = undefined,
        .execution_mode = undefined,
    };

    const id_count = iterator.module[3];

    const Id = struct {
        opcode: u32,
        type_id: u32,
        storage_class: u32,
        binding: u32,
        set: u32,
        constant: u32,
        array_length: u32,
    };

    var ids = try allocator.alloc(Id, id_count);
    defer allocator.free(ids);

    for (ids) |*id| {
        id.array_length = 1;
    }

    while (iterator.next()) |op_data| {
        switch (op_data.op) {
            .EntryPoint => {
                std.log.info("Found shader entry point", .{});

                const name_begin = @as([*:0]const u8, @ptrCast(&op_data.words[3]));

                std.log.info("{c}", .{name_begin[1]});

                const name = std.mem.span(name_begin);

                result.entry_point = name;
            },
            .Decorate => {},
            .TypeStruct,
            .TypeImage,
            .TypeSampler,
            .TypeSampledImage,
            .TypeArray,
            .TypeRuntimeArray,
            => {},
            .TypePointer => {},
            .Constant => {},
            .Variable => {},
            else => {},
        }
    }
}

test "parse" {
    try parse(std.testing.allocator, @alignCast(@embedFile("../test.spv")));
}

const std = @import("std");
const spirv = @import("../spirv.zig");
const Iterator = @import("Iterator.zig");
