pub fn lowerFromSpirvAir(
    allocator: std.mem.Allocator,
    air: spirv.Air,
) !Mir {
    var mir_instructions: std.ArrayListUnmanaged(Mir.Instruction) = .{};
    defer mir_instructions.deinit(allocator);

    for (air.functions) |function| {
        for (air.blocks[function.start_block..function.end_block]) |block| {
            for (air.instructions[block.start_instruction..block.end_instruction]) |instruction| {
                switch (instruction.tag) {
                    .load => {
                        try mir_instructions.append(allocator, .{ .tag = .mov });
                    },
                    .store => {
                        try mir_instructions.append(allocator, .{ .tag = .mov });
                    },
                    .matrix_vector_mul => {
                        for (0..4) |n| {
                            _ = n;
                            try mir_instructions.append(allocator, .{
                                .tag = .vdpps,
                                .operands = .{
                                    .{ .register = .xmm0 },
                                    .{ .register = .xmm1 },
                                    .{ .immediate = 0xff },
                                    .none,
                                },
                            });
                        }
                    },
                    .@"return" => {
                        try mir_instructions.append(allocator, .{ .tag = .ret });
                    },
                    else => {
                        try mir_instructions.append(allocator, .{ .tag = .nop });
                    },
                }
            }
        }
    }

    var mir: Mir = .{ .instructions = try mir_instructions.toOwnedSlice(allocator) };

    return mir;
}

const std = @import("std");
const spirv = @import("../spirv.zig");
const Mir = @import("Mir.zig");
