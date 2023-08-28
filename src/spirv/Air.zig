//! The abstract intermediate representation for a spirv module

///Should be an array?
capability: spirv.Capability,
addressing_mode: spirv.AddressingMode,
memory_model: spirv.MemoryModel,
entry_point: struct {
    execution_mode: spirv.ExecutionModel,
    name: []const u8,
    interface: []const spirv.WordInt,
},
instructions: []Instruction,
blocks: []Block,
functions: []Function,
variables: []Variable,
types: []Type,

pub fn fromSpirvBytes(allocator: std.mem.Allocator, bytes: []align(4) const u8) !Air {
    var iterator = spirv.Iterator.initFromSlice(bytes);

    var instructions: std.ArrayListUnmanaged(Instruction) = .{};
    defer instructions.deinit(allocator);

    var blocks: std.ArrayListUnmanaged(Block) = .{};
    defer blocks.deinit(allocator);

    var functions: std.ArrayListUnmanaged(Function) = .{};
    defer functions.deinit(allocator);

    var variables: std.ArrayListUnmanaged(Variable) = .{};
    _ = variables;

    var types: std.ArrayListUnmanaged(Type) = .{};
    defer types.deinit(allocator);

    var type_index_map: TypeIndexMap = .{};
    defer type_index_map.deinit(allocator);

    var op_result_indices = try std.ArrayListUnmanaged(u32).initCapacity(allocator, iterator.module.len);
    defer op_result_indices.deinit(allocator);

    try op_result_indices.appendNTimes(allocator, std.math.maxInt(u32), iterator.module.len);

    var air: Air = .{
        .capability = undefined,
        .memory_model = undefined,
        .addressing_mode = undefined,
        .entry_point = undefined,
        .instructions = &.{},
        .blocks = &.{},
        .functions = &.{},
        .variables = &.{},
        .types = &.{},
    };

    var id_op_index: u32 = 5;

    while (iterator.next()) |op| {
        defer id_op_index += 1;

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
            .Capability => {
                air.capability = @enumFromInt(op.words[1]);
            },
            .MemoryModel => {
                air.addressing_mode = @enumFromInt(op.words[1]);
                air.memory_model = @enumFromInt(op.words[2]);
            },
            else => {},
        }

        const result_index: u32 = switch (op.op) {
            .ExtInstImport => 2,
            .TypeVoid => 1,
            .TypeBool => 2,
            .TypeInt => 2,
            .TypeFloat => 2,
            .TypeVector => 2,
            .TypeMatrix => 2,
            .TypeImage => 2,
            .TypeSampler => 2,
            .TypeSampledImage => 2,
            .TypeArray => 2,
            .TypeRuntimeArray => 2,
            .TypeStruct => 2,
            .TypeOpaque => 2,
            .TypePointer => 2,
            .TypeFunction => 2,
            .TypeEvent => 2,
            .TypeDeviceEvent => 2,
            .TypeReserveId => 2,
            .TypeQueue => 2,
            .TypePipe => 2,
            .TypeForwardPointer => 2,
            .Variable => 2,
            .Constant => 2,
            .Label => 1,
            else => 0,
        };

        const has_result = result_index != 0;

        if (op.words[result_index] >= op_result_indices.items.len) {
            continue;
        }

        std.log.info("lol op = {}, res indx {}", .{ op.op, result_index });
        if (has_result) op_result_indices.items[op.words[result_index]] = id_op_index;
    }

    iterator = spirv.Iterator.initFromSlice(bytes);

    var op_index: u32 = 0;

    while (iterator.next()) |op| {
        defer op_index += 1;

        std.log.info("op: {}", .{op.op});

        switch (op.op) {
            .Function => {
                // const return_type = try parseType(
                //     allocator,
                //     bytes,
                //     &types,
                //     &type_index_map,
                //     op_result_indices.items[op.words[2]],
                // );
                // _ = return_type;

                try functions.append(
                    allocator,
                    .{
                        .start_block = @intCast(blocks.items.len),
                        .end_block = @intCast(blocks.items.len),
                        .return_type = 0,
                        .function_control = @bitCast(op.words[2]),
                        .type_function = 0,
                    },
                );
            },
            .FunctionEnd => {
                functions.items[functions.items.len - 1].end_block = @intCast(blocks.items.len);
            },
            .Label => {
                if (blocks.items.len > 0) {
                    blocks.items[blocks.items.len - 1].end_instruction = @intCast(instructions.items.len);
                }

                try blocks.append(
                    allocator,
                    .{
                        .start_instruction = @intCast(instructions.items.len),
                        .end_instruction = @intCast(instructions.items.len),
                    },
                );
            },
            .Variable => {
                try instructions.append(allocator, .{ .tag = .allocate_variable });
            },
            .Load => {
                try instructions.append(allocator, .{ .tag = .load });
            },
            .Store => {
                try instructions.append(allocator, .{ .tag = .store });
            },
            .CompositeExtract => {
                try instructions.append(allocator, .{ .tag = .composite_extract });
            },
            .CompositeConstruct => {
                try instructions.append(allocator, .{ .tag = .composite_construct });
            },
            .AccessChain => {
                try instructions.append(allocator, .{ .tag = .access_chain });
            },
            .MatrixTimesVector => {
                try instructions.append(allocator, .{ .tag = .matrix_vector_mul });
            },
            .Return => {
                try instructions.append(allocator, .{ .tag = .@"return" });

                if (blocks.items.len > 0) {
                    blocks.items[blocks.items.len - 1].end_instruction = @intCast(instructions.items.len);
                }
            },
            else => {},
        }
    }

    air.instructions = try instructions.toOwnedSlice(allocator);
    air.blocks = try blocks.toOwnedSlice(allocator);
    air.functions = try functions.toOwnedSlice(allocator);
    air.types = try types.toOwnedSlice(allocator);

    return air;
}

pub fn deinit(self: *Air, allocator: std.mem.Allocator) void {
    defer self.* = undefined;

    allocator.free(self.instructions);
    allocator.free(self.blocks);
    allocator.free(self.functions);
    allocator.free(self.types);
    allocator.free(self.variables);
}

pub const InstructionIndex = u32;

///Convient representation of an operation
///This should be thought of as a "fat" IR for spirv
///Not all are ops supported yet
///This is an instruction that is an operation, so
///Functions, types, debug info and general module info is not
///Stored as an 'instruction'
pub const Instruction = struct {
    tag: Tag,

    pub const Tag = enum {
        allocate_variable,
        branch,
        branch_if,
        @"switch",
        load,
        store,
        composite_construct,
        composite_extract,
        access_chain,
        matrix_vector_mul,
        matrix_matrix_mul,
        @"return",
    };
};

pub fn parseType(
    allocator: std.mem.Allocator,
    bytes: []align(4) const u8,
    types: *std.ArrayListUnmanaged(Type),
    type_index_map: *TypeIndexMap,
    op_index: u32,
) !TypeIndex {
    const existing_index = type_index_map.get(op_index);

    if (existing_index != null) return existing_index.?;

    std.log.info("op_index = {}", .{op_index});

    var iterator = spirv.Iterator.initFromSlice(@alignCast(bytes[(op_index) * 4 ..]));

    const @"type": Type = if (iterator.next()) |op| switch (op.op) {
        .TypeVoid => .{ .primitive = .void },
        .TypeBool => .{ .primitive = .bool },
        .TypeInt => unreachable,
        .TypeFloat => unreachable,
        .TypeVector => unreachable,
        .TypeMatrix => unreachable,
        .TypeImage => unreachable,
        .TypeSampler => unreachable,
        .TypeSampledImage => unreachable,
        .TypeArray => unreachable,
        .TypeRuntimeArray => unreachable,
        .TypeStruct => unreachable,
        .TypeOpaque => unreachable,
        .TypePointer => unreachable,
        .TypeFunction => unreachable,
        .TypeEvent => unreachable,
        .TypeDeviceEvent => unreachable,
        .TypeReserveId => unreachable,
        .TypeQueue => unreachable,
        .TypePipe => unreachable,
        .TypeForwardPointer => unreachable,
        else => {
            std.log.info("[{}]/[{}] type op = {}", .{ op_index, op_index / 4, op.op });

            unreachable;
        },
    } else unreachable;

    const type_index: TypeIndex = @intCast(types.items.len - 1);

    try types.append(allocator, @"type");

    type_index_map.put(allocator, op_index, type_index) catch unreachable;

    return type_index;
}

const TypeIndexMap = std.AutoArrayHashMapUnmanaged(u32, TypeIndex);

pub const BlockIndex = u32;

///Basic block of instructions
pub const Block = struct {
    start_instruction: InstructionIndex,
    end_instruction: InstructionIndex,
};

pub const FunctionIndex = u32;

pub const Function = struct {
    start_block: BlockIndex,
    end_block: BlockIndex,
    return_type: TypeIndex,
    function_control: spirv.FunctionControl,
    type_function: spirv.WordInt,
};

pub const VariableIndex = u32;

pub const Variable = struct {
    type: TypeIndex,
    storage_class: spirv.StorageClass,
    initializer: Initializer,

    pub const Initializer = union(enum) {
        constant: void,
        instruction: InstructionIndex,
    };
};

pub const TypeIndex = u32;

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
    members: []TypeIndex,
};

pub const TypeArray = struct {
    child_type: TypeIndex,
    len: u32,
};

const std = @import("std");
const spirv = @import("../spirv.zig");
const Air = @This();
