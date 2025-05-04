pub const Result = struct {
    entry_point: [:0]const u8,
    source_language: spirv.SourceLanguage,
    execution_model: spirv.ExecutionModel,
    addressing_mode: spirv.AddressingMode,
    memory_model: spirv.MemoryModel,
    execution_mode: spirv.ExecutionMode,

    local_size_x: u32 = 0,
    local_size_y: u32 = 0,
    local_size_z: u32 = 0,
    resources: [32]Resource = std.mem.zeroes([32]Resource),
    resource_count: u32 = 0,
    resource_mask: u32 = 0,

    const Resource = struct {
        descriptor_type: spirv.Op,
        descriptor_count: u32,
        binding: u32,
    };
};

pub fn parse(allocator: std.mem.Allocator, result: *Result, module: []align(4) const u8) !void {
    var iterator = Iterator.initFromSlice(module);

    std.debug.assert(iterator.module[0] == spirv.magic_number);

    const id_count = iterator.module[3];

    const Id = struct {
        opcode: spirv.Op,
        type_id: u32,
        storage_class: spirv.StorageClass,
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
                const name_begin = @as([*:0]const u8, @ptrCast(&op_data.words[3]));

                const name = std.mem.span(name_begin);

                result.entry_point = name;
            },
            .Decorate => {
                const id = op_data.words[1];

                std.debug.assert(id < id_count);

                switch (@as(spirv.Decoration, @enumFromInt(op_data.words[2]))) {
                    .DescriptorSet => {
                        ids[id].set = op_data.words[3];
                    },
                    .Binding => {
                        ids[id].binding = op_data.words[3];
                    },
                    else => {},
                }
            },
            .TypeStruct,
            .TypeImage,
            .TypeSampler,
            .TypeSampledImage,
            .TypeArray,
            .TypeRuntimeArray,
            => {
                const id = op_data.words[1];

                ids[id].opcode = op_data.op;

                switch (op_data.op) {
                    .TypeArray => {
                        ids[id].opcode = ids[op_data.words[2]].opcode;
                        ids[id].array_length = ids[op_data.words[3]].constant;
                    },
                    else => {},
                }
            },
            .TypePointer => {
                const id = op_data.words[1];

                ids[id].opcode = op_data.op;
                ids[id].type_id = op_data.words[3];
                ids[id].storage_class = @as(spirv.StorageClass, @enumFromInt(op_data.words[2]));
            },
            .Constant => {
                const id = op_data.words[2];

                ids[id].opcode = op_data.op;
                ids[id].type_id = op_data.words[1];
                ids[id].constant = op_data.words[3];
            },
            .Variable => {
                const id = op_data.words[2];

                ids[id].opcode = op_data.op;
                ids[id].type_id = op_data.words[1];
                ids[id].storage_class = @as(spirv.StorageClass, @enumFromInt(op_data.words[3]));
            },
            else => {},
        }
    }

    id_loop: for (ids) |id| {
        if (id.opcode == .Variable and
            (id.storage_class == .Uniform or
                id.storage_class == .UniformConstant or
                id.storage_class == .StorageBuffer or
                id.storage_class == .Image))
        {
            const type_kind = ids[ids[id.type_id].type_id].opcode;
            const array_length = ids[ids[id.type_id].type_id].array_length;
            const resource_type = type_kind;

            {
                for (result.resources) |resource| {
                    if (resource.binding == id.binding and resource.descriptor_type == resource_type) {
                        continue :id_loop;
                    }
                }
            }

            result.resources[result.resource_count] = .{
                .descriptor_type = resource_type,
                .descriptor_count = array_length,
                .binding = id.binding,
            };

            result.resource_mask |= @as(u32, 1) << @as(u5, @intCast(id.binding));
            result.resource_count += 1;
        }
    }
}

test "parse" {
    // try parse(std.testing.allocator, @alignCast(@embedFile("../test.spv")));
}

const std = @import("std");
const spirv = @import("../spirv.zig");
const Iterator = @import("Iterator.zig");
