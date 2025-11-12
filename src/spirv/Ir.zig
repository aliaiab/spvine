//! A sea of nodes representation for spirv semantics

node_buffer: std.ArrayList(u32) = .{},
effect_nodes: std.ArrayList(Node) = .{},
node_list: std.ArrayList(Node) = .{},
types: std.AutoArrayHashMapUnmanaged(Node, void) = .{},
constants: std.AutoArrayHashMapUnmanaged(Node.Constant, Node) = .{},
node_dedup_map: std.StringArrayHashMapUnmanaged(Node) = .{},
node_scheduled_map: std.AutoArrayHashMapUnmanaged(Node, void) = .{},
optimization_flags: OptimizationFlags = .{},

pub const OptimizationFlags = packed struct(u32) {
    enable_constant_folding: bool = false,
    enable_constant_hoisting: bool = false,
    _: u30 = 0,
};

pub fn buildNodeOpConstant(
    ir: *Ir,
    allocator: std.mem.Allocator,
    constant: Node.Constant,
) !Node {
    return try ir.buildNode(allocator, .constant, Node.Constant, constant);
}

pub fn buildNodeOpVariable(
    ir: *Ir,
    allocator: std.mem.Allocator,
    result_type: Node,
    storage_class: spirv.StorageClass,
    initializer: Node,
) !Node {
    return try ir.buildNode(allocator, .variable, Node.Variable, .{
        .type = result_type,
        .storage_class = storage_class,
        .initializer = initializer,
    });
}

pub fn buildNodeOpLoad(
    ir: *Ir,
    allocator: std.mem.Allocator,
    result_type: Node,
    pointer: Node,
) !Node {
    return try ir.buildNode(allocator, .load, Node.Load, .{
        .result_type = result_type,
        .pointer = pointer,
    });
}

pub fn buildNodeOpConvertSToF(
    ir: *Ir,
    allocator: std.mem.Allocator,
    result_type: Node,
    operand: Node,
) !Node {
    if (ir.optimization_flags.enable_constant_folding) {
        if (ir.nodeTag(operand).* == .constant) {
            const value = ir.nodeData(operand, Node.Constant);

            const value_float: f32 = @floatFromInt(value.value_bits);

            return try ir.buildNodeOpConstant(allocator, .{
                .type = result_type,
                .value_bits = @bitCast(value_float),
            });
        }
    }

    return try ir.buildNode(allocator, .convert_s_to_f, Node.ConvertSToF, .{
        .type = result_type,
        .operand = operand,
    });
}

pub fn buildNodeOpIAdd(
    ir: *Ir,
    allocator: std.mem.Allocator,
    result_type: Node,
    input_lhs: Node,
    input_rhs: Node,
) !Node {
    const lhs = input_lhs;
    const rhs = input_rhs;

    if (ir.optimization_flags.enable_constant_folding) {
        if (ir.nodeTag(lhs).* == .constant and ir.nodeTag(rhs).* == .constant) {
            const lhs_value = ir.nodeData(lhs, Node.Constant);
            const rhs_value = ir.nodeData(rhs, Node.Constant);

            const addition = lhs_value.value(i32) + rhs_value.value(i32);

            return try ir.buildNode(allocator, .constant, Node.Constant, .{
                .type = lhs_value.type,
                .value_bits = @bitCast(addition),
            });
        }
    }

    const result_node = try ir.buildNode(allocator, .iadd, Node.IAdd, .{
        .type = result_type,
        .lhs = lhs,
        .rhs = rhs,
    });

    return result_node;
}

pub fn buildNodeOpFAdd(
    ir: *Ir,
    allocator: std.mem.Allocator,
    result_type: Node,
    input_lhs: Node,
    input_rhs: Node,
) !Node {
    var lhs = input_lhs;
    var rhs = input_rhs;

    if (ir.optimization_flags.enable_constant_hoisting) {
        //Canonicalize constant/non-constant pairs so the constant always appears on the left
        //Boolean xor
        if ((ir.nodeTag(lhs).* == .constant) != (ir.nodeTag(rhs).* == .constant)) {
            if (ir.nodeTag(rhs).* == .constant) {
                std.mem.swap(Node, &lhs, &rhs);
            }
        }
    }

    if (ir.optimization_flags.enable_constant_hoisting) {
        //Hoists non-consequetives constants that share a common operation
        //Rewrites the graph in the following way:

        //add/add:
        // 2 + (3 + a) -> (2 + 3) + a

        //add/sub:
        //2 + (3 - a) -> 2 + 3 - a

        //mul/mul
        // 2 * (3 * a) -> (2 * 3) * a

        //mul/div (only for floats)
        // 2 * (3 / a) -> (2 * 3) / a

        //div/mul
        // 2 / (3 * a) -> (2 / 3) / a

        //negate/mul
        //-(3 + a) -> -3 - a

        if (ir.nodeTag(lhs).* == .constant) {
            const rhs_tag = ir.nodeTag(rhs).*;

            switch (rhs_tag) {
                .fadd => {
                    const rhs_add = ir.nodeData(rhs, Node.FAdd);

                    if ((ir.nodeTag(rhs_add.lhs).* == .constant) != (ir.nodeTag(rhs_add.rhs).* == .constant)) {
                        const rhs_add_const = rhs_add.lhs;

                        const constant = try ir.buildNodeOpFAdd(
                            allocator,
                            result_type,
                            lhs,
                            rhs_add_const,
                        );

                        lhs = constant;
                        rhs = rhs_add.rhs;
                    }
                },
                .fsub => {
                    const rhs_sub = ir.nodeData(rhs, Node.FSub);

                    if ((ir.nodeTag(rhs_sub.lhs).* == .constant) != (ir.nodeTag(rhs_sub.rhs).* == .constant)) {
                        const rhs_add_const = rhs_sub.lhs;

                        const constant = try ir.buildNodeOpFAdd(
                            allocator,
                            result_type,
                            lhs,
                            rhs_add_const,
                        );

                        lhs = constant;
                        rhs = rhs_sub.rhs;

                        return try ir.buildNodeOpFSub(
                            allocator,
                            result_type,
                            lhs,
                            rhs,
                        );
                    }
                },
                else => {},
            }
        }
    }

    if (ir.optimization_flags.enable_constant_folding) {
        if (ir.nodeTag(lhs).* == .constant and ir.nodeTag(rhs).* == .constant) {
            const lhs_value = ir.nodeData(lhs, Node.Constant);
            const rhs_value = ir.nodeData(rhs, Node.Constant);

            const addition = lhs_value.value(f32) + rhs_value.value(f32);

            return try ir.buildNode(allocator, .constant, Node.Constant, .{
                .type = lhs_value.type,
                .value_bits = @bitCast(addition),
            });
        }
    }

    const result_node = try ir.buildNode(allocator, .fadd, Node.FAdd, .{
        .type = result_type,
        .lhs = lhs,
        .rhs = rhs,
    });

    return result_node;
}

pub fn buildNodeOpFSub(
    ir: *Ir,
    allocator: std.mem.Allocator,
    result_type: Node,
    lhs: Node,
    rhs: Node,
) !Node {
    if (ir.optimization_flags.enable_constant_folding) {
        if (ir.nodeTag(lhs).* == .constant and ir.nodeTag(rhs).* == .constant) {
            const lhs_value = ir.nodeData(lhs, Node.Constant);
            const rhs_value = ir.nodeData(rhs, Node.Constant);

            const addition = lhs_value.value(f32) - rhs_value.value(f32);

            return try ir.buildNode(allocator, .constant, Node.Constant, .{
                .type = lhs_value.type,
                .value_bits = @bitCast(addition),
            });
        }
    }

    const result_node = try ir.buildNode(allocator, .fsub, Node.FSub, .{
        .type = result_type,
        .lhs = lhs,
        .rhs = rhs,
    });

    return result_node;
}

pub fn buildNodeOpFMul(
    ir: *Ir,
    allocator: std.mem.Allocator,
    result_type: Node,
    lhs: Node,
    rhs: Node,
) !Node {
    if (ir.optimization_flags.enable_constant_folding) {
        if (ir.nodeTag(lhs).* == .constant and ir.nodeTag(rhs).* == .constant) {
            const lhs_value = ir.nodeData(lhs, Node.Constant);
            const rhs_value = ir.nodeData(rhs, Node.Constant);

            const addition = lhs_value.value(f32) * rhs_value.value(f32);

            return try ir.buildNode(allocator, .constant, Node.Constant, .{
                .type = lhs_value.type,
                .value_bits = @bitCast(addition),
            });
        }
    }

    const result_node = try ir.buildNode(allocator, .fmul, Node.FMul, .{
        .type = result_type,
        .lhs = lhs,
        .rhs = rhs,
    });

    return result_node;
}

pub fn buildNodeOpFNegate(
    ir: *Ir,
    allocator: std.mem.Allocator,
    result_type: Node,
    operand: Node,
) !Node {
    if (ir.optimization_flags.enable_constant_folding) {
        if (ir.nodeTag(operand).* == .constant) {
            const operand_value = ir.nodeData(operand, Node.Constant);

            const negation = -operand_value.value(f32);

            return try ir.buildNode(allocator, .constant, Node.Constant, .{
                .type = operand_value.type,
                .value_bits = @bitCast(negation),
            });
        }
    }

    const result_node = try ir.buildNode(allocator, .fnegate, Node.FNegate, .{
        .type = result_type,
        .operand = operand,
    });

    return result_node;
}

///Build a generic node
pub fn buildNode(
    ir: *Ir,
    allocator: std.mem.Allocator,
    node_tag: Node.Tag,
    T: type,
    node_data: T,
) !Node {
    const instruction = &node_data;

    const node_bytes = std.mem.asBytes(instruction);

    const node_query = try ir.node_dedup_map.getOrPut(allocator, node_bytes);

    //Some nodes must be considered distinct, like op_variable
    const can_deduplicate = switch (node_tag) {
        .variable => false,
        .return_value => false,
        else => true,
    };

    if (node_query.found_existing and can_deduplicate) {
        return node_query.value_ptr.*;
    }

    const words_to_append = std.math.divCeil(comptime_int, @sizeOf(T), @sizeOf(u32)) catch @compileError("Type T have size divisible by 4 bytes");

    const node_offset: usize = ir.node_buffer.items.len;
    try ir.node_buffer.appendNTimes(allocator, undefined, words_to_append);

    const node: Node = .{ .offset = @intCast(node_offset) };

    ir.nodeData(node, T).* = node_data;

    switch (node_tag) {
        .return_value => {
            try ir.effect_nodes.append(allocator, node);
        },
        else => {},
    }

    node_query.value_ptr.* = node;

    return node;
}

pub fn nodeTag(ir: Ir, node: Node) *Node.Tag {
    const ptr: [*]u8 = @ptrCast(ir.node_buffer.items.ptr + node.offset);

    return @ptrCast(@alignCast(ptr));
}

pub fn nodeData(ir: Ir, node: Node, comptime T: type) *T {
    const ptr: [*]u8 = @ptrCast(ir.node_buffer.items.ptr + node.offset);

    return @ptrCast(@alignCast(ptr));
}

pub fn computeGlobalOrdering(
    ir: *Ir,
    ordering: *std.ArrayList(Node),
    allocator: std.mem.Allocator,
) !void {
    for (ir.effect_nodes.items) |effect_node| {
        try ir.computeGlobalOrderingForNode(effect_node, ordering, allocator);
    }
}

///Performs global code motion to determine structure total ordering
pub fn computeGlobalOrderingForNode(
    ir: *Ir,
    root_node: Node,
    ordering: *std.ArrayList(Node),
    allocator: std.mem.Allocator,
) !void {
    if (ir.node_scheduled_map.get(root_node)) |_| {
        return;
    }

    try ir.node_scheduled_map.put(allocator, root_node, {});

    if (root_node == Node.nil) {
        return;
    }

    const OrderResult = struct {
        node: Node,
        //The value number/order for the node
        order: u32,
    };

    var order_stack: std.ArrayList(OrderResult) = .{};
    defer order_stack.clearAndFree(allocator);

    switch (ir.nodeTag(root_node).*) {
        .null,
        .return_value,
        => {
            const return_value = ir.nodeData(root_node, Node.ReturnValue);

            try ir.computeGlobalOrderingForNode(return_value.return_value, ordering, allocator);

            try ordering.append(allocator, root_node);
        },
        .type_int,
        .type_float,
        => {
            try ordering.append(allocator, root_node);
        },
        .type_pointer => {
            const instruction = ir.nodeData(root_node, Node.TypePointer);

            try ir.computeGlobalOrderingForNode(
                instruction.type,
                ordering,
                allocator,
            );

            try ordering.append(allocator, root_node);
        },
        .constant => {
            const variable = ir.nodeData(root_node, Node.Constant);

            try ir.computeGlobalOrderingForNode(
                variable.type,
                ordering,
                allocator,
            );

            try ordering.append(allocator, root_node);
        },
        .variable => {
            const variable = ir.nodeData(root_node, Node.Variable);

            try ir.computeGlobalOrderingForNode(
                variable.type,
                ordering,
                allocator,
            );

            try ir.computeGlobalOrderingForNode(
                variable.initializer,
                ordering,
                allocator,
            );

            try ordering.append(allocator, root_node);
        },
        .load => {
            const instruction = ir.nodeData(root_node, Node.Load);

            try ir.computeGlobalOrderingForNode(
                instruction.pointer,
                ordering,
                allocator,
            );

            try ordering.append(allocator, root_node);
        },
        .convert_s_to_f => {
            const instruction = ir.nodeData(root_node, Node.ConvertSToF);

            try ir.computeGlobalOrderingForNode(instruction.operand, ordering, allocator);

            try ordering.append(allocator, root_node);
        },
        .iadd => {
            const fadd = ir.nodeData(root_node, Node.IAdd);

            try ir.computeGlobalOrderingForNode(fadd.lhs, ordering, allocator);
            try ir.computeGlobalOrderingForNode(fadd.rhs, ordering, allocator);

            try ordering.append(allocator, root_node);
        },
        .fadd => {
            const fadd = ir.nodeData(root_node, Node.FAdd);

            try ir.computeGlobalOrderingForNode(fadd.lhs, ordering, allocator);
            try ir.computeGlobalOrderingForNode(fadd.rhs, ordering, allocator);

            try ordering.append(allocator, root_node);
        },
        .fsub => {
            const fsub = ir.nodeData(root_node, Node.FSub);

            try ir.computeGlobalOrderingForNode(fsub.lhs, ordering, allocator);
            try ir.computeGlobalOrderingForNode(fsub.rhs, ordering, allocator);

            try ordering.append(allocator, root_node);
        },
        .fmul => {
            const fmul = ir.nodeData(root_node, Node.FAdd);

            try ir.computeGlobalOrderingForNode(fmul.lhs, ordering, allocator);
            try ir.computeGlobalOrderingForNode(fmul.rhs, ordering, allocator);

            try ordering.append(allocator, root_node);
        },
        .fnegate => {
            const fnegate = ir.nodeData(root_node, Node.FNegate);

            try ir.computeGlobalOrderingForNode(fnegate.operand, ordering, allocator);

            try ordering.append(allocator, root_node);
        },
    }
}

pub const Node = packed struct(u32) {
    offset: u32,

    pub const nil: Node = .{ .offset = std.math.maxInt(u32) };

    pub const Tag = enum(u32) {
        null,

        type_int,
        type_float,
        type_pointer,
        constant,

        variable,
        load,
        iadd,
        convert_s_to_f,
        fadd,
        fsub,
        fmul,
        fnegate,
        return_value,
    };

    pub const Data = union(Tag) {
        null: void,
        type_int: TypeInt,
        type_float: TypeFloat,
        type_pointer: TypePointer,
        constant: Constant,
        load: Load,
        variable: Variable,
        iadd: IAdd,
        convert_s_to_f: ConvertSToF,
        fadd: FAdd,
        fsub: FSub,
        fmul: FMul,
        fnegate: FNegate,
        return_value: ReturnValue,
    };

    pub const TypeFloat = struct {
        tag: Tag = .type_float,
        bit_width: u32,
    };

    pub const TypeInt = struct {
        tag: Tag = .type_int,
        bit_width: u16,
        signedness: Signedness,
        range_value_min: u32,
        range_value_max: u32,

        pub const Signedness = enum(u16) {
            unsigned,
            signed,
        };
    };

    pub const TypePointer = struct {
        tag: Tag = .type_pointer,
        type: Node,
        storage_class: spirv.StorageClass,
    };

    pub const Load = struct {
        tag: Tag = .load,
        result_type: Node,
        pointer: Node,
    };

    ///A scalar constant
    pub const Constant = struct {
        tag: Tag = .constant,
        type: Node,
        value_bits: u32,

        pub fn value(constant: Constant, comptime T: type) T {
            return @bitCast(constant.value_bits);
        }
    };

    pub const Variable = struct {
        tag: Tag = .variable,
        type: Node,
        initializer: Node,
        storage_class: spirv.StorageClass,
    };

    pub const IAdd = struct {
        tag: Tag = .iadd,
        type: Node,
        lhs: Node,
        rhs: Node,
    };

    pub const ConvertSToF = struct {
        tag: Tag = .convert_s_to_f,
        type: Node,
        operand: Node,
    };

    pub const FAdd = struct {
        tag: Tag = .fadd,
        type: Node,
        lhs: Node,
        rhs: Node,
    };

    pub const FSub = struct {
        tag: Tag = .fsub,
        type: Node,
        lhs: Node,
        rhs: Node,
    };

    pub const FMul = struct {
        tag: Tag = .fmul,
        type: Node,
        lhs: Node,
        rhs: Node,
    };

    pub const FNegate = struct {
        tag: Tag = .fnegate,
        type: Node,
        operand: Node,
    };

    pub const ReturnValue = struct {
        tag: Tag = .return_value,
        return_value: Node,
    };
};

pub fn printNodes(ir: Ir, writer: *std.Io.Writer) !void {
    _ = try writer.print("Types:\n\n", .{});

    for (ir.types.keys()) |node| {
        _ = try writer.splatByteAll(' ', 4);

        switch (ir.nodeTag(node).*) {
            .null => {},
            .type_int => {
                const type_int = ir.nodeData(node, Node.TypeInt);

                try writer.print("$0x{x:0>2} := type_int: bit_width: {}\n", .{ node.offset, type_int.bit_width });
            },
            .type_float => {
                const type_float = ir.nodeData(node, Node.TypeFloat);

                try writer.print("$0x{x:0>2} := type_float: bit_width: {}\n", .{ node.offset, type_float.bit_width });
            },
            .type_pointer => {
                const type_float = ir.nodeData(node, Node.TypePointer);

                try writer.print("$0x{x:0>2} := type_pointer: type: %0x{x}, storage_class: {s}\n", .{
                    node.offset,
                    type_float.type.offset,
                    @tagName(type_float.storage_class),
                });
            },
            else => {},
        }
    }

    _ = try writer.print("\nConstants:\n\n", .{});

    for (ir.constants.values()) |node| {
        _ = try writer.splatByteAll(' ', 4);

        switch (ir.nodeTag(node).*) {
            else => {},
        }
    }

    _ = try writer.print("\nInstructions:\n\n", .{});

    for (ir.node_list.items) |node| {
        _ = try writer.splatByteAll(' ', 4);

        switch (ir.nodeTag(node).*) {
            .null => {},
            .constant => {
                const constant = ir.nodeData(node, Node.Constant);

                try writer.print("$0x{x:0>2} := op_constant: type: $0{x}, value: ", .{ node.offset, constant.type.offset });

                try ir.printOperand(writer, node);

                try writer.print("\n", .{});
            },
            .type_int => {
                const type_int = ir.nodeData(node, Node.TypeInt);

                try writer.print("$0x{x:0>2} := type_int: bit_width: {}\n", .{ node.offset, type_int.bit_width });
            },
            .type_float => {
                const type_float = ir.nodeData(node, Node.TypeFloat);

                try writer.print("$0x{x:0>2} := type_float: bit_width: {}\n", .{ node.offset, type_float.bit_width });
            },
            .type_pointer => {
                const type_float = ir.nodeData(node, Node.TypePointer);

                try writer.print("$0x{x:0>2} := type_pointer: type: %0x{x}, storage_class: {s}\n", .{
                    node.offset,
                    type_float.type.offset,
                    @tagName(type_float.storage_class),
                });
            },
            .variable => {
                const variable = ir.nodeData(node, Node.Variable);
                _ = variable; // autofix

                try writer.print("$0x{x:0>2} := op_variable\n", .{node.offset});
            },
            .load => {
                const instruction = ir.nodeData(node, Node.Load);

                try writer.print("$0x{x:0>2} := op_load: ", .{node.offset});

                try ir.printOperand(writer, instruction.pointer);

                try writer.writeAll("\n");
            },
            .iadd => {
                const instruction = ir.nodeData(node, Node.IAdd);

                try writer.print("$0x{x:0>2} := op_iadd: ", .{node.offset});

                try ir.printOperand(writer, instruction.lhs);

                try writer.print(", ", .{});

                try ir.printOperand(writer, instruction.rhs);

                try writer.writeAll("\n");
            },
            .convert_s_to_f => {
                const instruction = ir.nodeData(node, Node.ConvertSToF);

                try writer.print("$0x{x:0>2} := op_convert_s_to_f: ", .{node.offset});

                try ir.printOperand(writer, instruction.operand);

                try writer.writeAll("\n");
            },
            .fadd => {
                const instruction = ir.nodeData(node, Node.FAdd);

                try writer.print("$0x{x:0>2} := op_fadd: ", .{node.offset});

                try ir.printOperand(writer, instruction.lhs);

                try writer.print(", ", .{});

                try ir.printOperand(writer, instruction.rhs);

                try writer.writeAll("\n");
            },
            .fsub => {
                const instruction = ir.nodeData(node, Node.FSub);

                try writer.print("$0x{x:0>2} := op_fsub: ", .{node.offset});

                try ir.printOperand(writer, instruction.lhs);

                try writer.print(", ", .{});

                try ir.printOperand(writer, instruction.rhs);

                try writer.writeAll("\n");
            },
            .fmul => {
                const fmul = ir.nodeData(node, Node.FAdd);

                try writer.print("$0x{x:0>2} := op_fmul: ", .{node.offset});

                try ir.printOperand(writer, fmul.lhs);

                try writer.print(", ", .{});

                try ir.printOperand(writer, fmul.rhs);

                try writer.writeAll("\n");
            },
            .fnegate => {
                const fnegate = ir.nodeData(node, Node.FNegate);

                try writer.print("$0x{x:0>2} := op_fnegate: ", .{node.offset});

                try ir.printOperand(writer, fnegate.operand);

                try writer.writeAll("\n");
            },
            .return_value => {
                const return_value = ir.nodeData(node, Node.ReturnValue);

                try writer.print("$0x{x:0>2} := op_return: ", .{node.offset});

                try ir.printOperand(writer, return_value.return_value);

                try writer.writeAll("\n");
            },
        }
    }
}

pub fn printOperand(ir: Ir, writer: *std.Io.Writer, node: Node) !void {
    switch (ir.nodeTag(node).*) {
        .constant => {
            const constant = ir.nodeData(node, Node.Constant);

            switch (ir.nodeTag(constant.type).*) {
                .type_int => {
                    const integer: u32 = constant.value_bits;

                    try writer.print("{}", .{integer});
                },
                .type_float => {
                    const float: f32 = @bitCast(constant.value_bits);

                    try writer.print("{:.1}", .{float});
                },
                else => {},
            }
        },
        else => {
            try writer.print("$0x{x}", .{node.offset});
        },
    }
}

const std = @import("std");
const Ir = @This();
const spirv = @import("../spirv.zig");
