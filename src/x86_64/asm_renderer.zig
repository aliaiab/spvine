//! Renders nasm assembly from Mir

pub fn print(mir: Mir, writer: anytype) !void {
    try writer.print("mir instruction count = {}\n\n", .{mir.instructions.len});

    try writer.print("vertex_main:\n", .{});

    for (mir.instructions) |instruction| {
        try writer.print("{s:>4}", .{" "});

        try writer.print("{s} ", .{@tagName(instruction.tag)});

        try renderOperands(writer, &instruction.operands);

        try writer.print("\n", .{});
    }
}

fn renderOperands(writer: anytype, operands: []const Mir.Operand) !void {
    for (operands, 0..) |operand, i| {
        switch (operand) {
            .immediate => |immediate| {
                try writer.print("0x{x}", .{immediate});
            },
            .register => |register| {
                try writer.print("{s}", .{@tagName(register)});
            },
            .none => break,
        }

        if (i == operands.len - 1) {
            break;
        }

        if (operands[i + 1] != .none) {
            try writer.print(", ", .{});
        }
    }
}

const std = @import("std");
const Mir = @import("Mir.zig");
