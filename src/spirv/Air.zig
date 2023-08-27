//! The abstract intermediate representation for a spirv module

entry_point: struct {
    execution_mode: spirv.ExecutionModel,
    name: []const u8,
    interface: []const spirv.WordInt,
},
instructions: []Instruction,
functions: []Function,
types: []Type,

pub fn fromSpirvBytes(allocator: std.mem.Allocator, bytes: []align(4) const u8) !Air {
    var iterator = spirv.Iterator.initFromSlice(bytes);

    var instructions: std.ArrayListUnmanaged(Instruction) = .{};
    defer instructions.deinit(allocator);

    var functions: std.ArrayListUnmanaged(Function) = .{};
    defer functions.deinit(allocator);

    var types: std.ArrayListUnmanaged(Type) = .{};
    defer types.deinit(allocator);

    var type_index_map: TypeIndexMap = .{};
    defer type_index_map.deinit(allocator);

    var air: Air = .{
        .entry_point = undefined,
        .instructions = &.{},
        .functions = &.{},
        .types = &.{},
    };

    var index: u32 = 0;
    var instruction_index: u32 = 0;

    while (iterator.next()) |op| {
        defer index +%= 1;

        std.log.info("op: {}", .{op.op});

        switch (op.op) {
            .EntryPoint => {
                const name: [:0]const u8 = std.mem.span(@as([*:0]const u8, @ptrCast(&op.words[3])));

                std.log.info("entry point name: {s}", .{name});

                air.entry_point = .{
                    .name = name,
                    .execution_mode = @enumFromInt(op.words[1]),
                    .interface = op.words[3 + (name.len / 4) + 1 ..],
                };
            },
            .Function => {
                const return_type = try parseType(
                    allocator,
                    bytes,
                    &types,
                    &type_index_map,
                    op.words[1],
                );

                try functions.append(
                    allocator,
                    .{
                        .start = instruction_index,
                        .end = instruction_index,
                        .return_type = return_type,
                        .function_control = @bitCast(op.words[2]),
                        .type_function = 0,
                    },
                );
            },
            .FunctionEnd => {
                functions.items[functions.items.len - 1].end = instruction_index;
            },
            else => {
                index -%= 1;
            },
        }
    }

    air.instructions = try instructions.toOwnedSlice(allocator);
    air.functions = try functions.toOwnedSlice(allocator);
    air.types = try types.toOwnedSlice(allocator);

    return air;
}

pub fn deinit(self: *Air, allocator: std.mem.Allocator) void {
    defer self.* = undefined;

    allocator.free(self.instructions);
    allocator.free(self.functions);
    allocator.free(self.types);
}

pub fn parseType(
    allocator: std.mem.Allocator,
    bytes: []align(4) const u8,
    types: *std.ArrayListUnmanaged(Type),
    type_index_map: *TypeIndexMap,
    index: u32,
) !Type {
    const exists = type_index_map.get(index) != null;

    if (exists) return types.items[index];

    defer type_index_map.put(allocator, index, {}) catch unreachable;

    var iterator = spirv.Iterator.initFromSlice(@alignCast(bytes[index / 4 ..]));

    while (iterator.next()) |op| {
        switch (op.op) {
            .TypeVoid => return .{ .primitive = .void },
            .TypeBool => return .{ .primitive = .bool },
            .TypeInt => {},
            .TypeFloat => {},
            .TypeVector => {},
            .TypeMatrix => {},
            .TypeImage => {},
            .TypeSampler => {},
            .TypeSampledImage => {},
            .TypeArray => {},
            .TypeRuntimeArray => {},
            .TypeStruct => {},
            .TypeOpaque => {},
            .TypePointer => {},
            .TypeFunction => {},
            .TypeEvent => {},
            .TypeDeviceEvent => {},
            .TypeReserveId => {},
            .TypeQueue => {},
            .TypePipe => {},
            .TypeForwardPointer => {},
            else => {},
        }

        return undefined;
    }

    unreachable;
}

///Convient representation of an operation
///This should be thought of as a "fat" IR for spirv
///Not all are ops supported yet
///This is an instruction that is an operation, so
///Functions, types, debug info and general module info is not
///Stored as an 'instruction'
pub const Instruction = union(enum) {
    constant: struct {
        type_id: spirv.WordInt,
        storage_class: spirv.StorageClass,
    },
    function: struct {
        return_type: Type,
        function_control: spirv.FunctionControl,
        type_function: spirv.WordInt,
    },
};

const TypeIndexMap = std.AutoArrayHashMapUnmanaged(u32, void);

pub const Function = struct {
    start: u32,
    end: u32,
    return_type: Type,
    function_control: spirv.FunctionControl,
    type_function: spirv.WordInt,
};

pub const Type = union(enum) {
    primitive: TypePrimitive,
    @"struct": TypeStruct,
    array: TypeArray,
};

pub const TypePrimitive = enum {
    void,
    bool,

    int8,
    int16,
    int32,
    int64,

    float16,
    float32,
    float64,

    vec2_f32,
    vec3_f32,
    vec4_f32,

    mat3x3_f32,
    mat4x4_f32,
    mat3x4_f32,
    mat4xx_f32,
};

pub const TypeStruct = struct {
    members: []Type,
};

pub const TypeArray = struct {
    child_type: *const Type,
    len: u32,
};

const std = @import("std");
const spirv = @import("../spirv.zig");
const Air = @This();
