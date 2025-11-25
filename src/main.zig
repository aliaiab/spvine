pub fn main() !void {
    var test_glsl_path: []const u8 = "src/test.glsl";

    var optimization_flags: spirv.Ir.OptimizationFlags = .{};
    var should_print_ast: bool = false;
    var should_print_ir: bool = false;

    {
        var args = std.process.args();

        _ = args.skip();

        while (args.next()) |opt_flag| {
            if (opt_flag[0] == '-') {
                if (std.mem.eql(u8, opt_flag, "-const_fold")) {
                    optimization_flags.enable_constant_folding = true;
                }

                if (std.mem.eql(u8, opt_flag, "-const_hoist")) {
                    optimization_flags.enable_constant_hoisting = true;
                }

                if (std.mem.eql(u8, opt_flag, "-debug_print_ast")) {
                    should_print_ast = true;
                }

                if (std.mem.eql(u8, opt_flag, "-debug_print_ir")) {
                    should_print_ir = true;
                }
            } else {
                const ext = std.fs.path.extension(opt_flag);

                if (std.mem.eql(u8, ext, ".glsl")) {
                    test_glsl_path = opt_flag;
                }
            }
        }
    }

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // defer std.debug.assert(gpa.deinit() != .leak);

    var allocator = gpa.allocator();

    if (@import("builtin").mode != .Debug) {
        allocator = std.heap.smp_allocator;
    }

    var stderr_buffer: [1024]u8 = undefined;
    var stderr_writer = std.fs.File.stderr().writer(&stderr_buffer);

    const stderr = &stderr_writer.interface;
    defer {
        stderr.flush() catch @panic("Flush error");
    }

    const file = try std.fs.cwd().openFile(test_glsl_path, .{});
    defer file.close();

    const test_glsl = try file.readToEndAlloc(allocator, std.math.maxInt(usize));
    defer allocator.free(test_glsl);

    var ast_node_arena_instance: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer ast_node_arena_instance.deinit();

    //Preinitialize the node arena so we don't hit gpa allocation very often if at all
    //TODO: figure out some sort of empirically based function based on the length of the source text to estimate the amount of node memory we need
    _ = try ast_node_arena_instance.allocator().alloc(u8, test_glsl.len * 10);
    _ = ast_node_arena_instance.reset(.retain_capacity);

    const ast_node_arena = ast_node_arena_instance.allocator();

    var ast = try Ast.parse(
        allocator,
        ast_node_arena,
        test_glsl,
        test_glsl_path,
    );
    defer ast.deinit(allocator);

    if (ast.errors.len != 0) {
        try glsl.error_render.printErrors(ast, null, ast.errors, stderr);

        return;
    }

    if (should_print_ast) {
        var unbuffered_stderr = std.fs.File.stderr().writer(&.{});

        try unbuffered_stderr.interface.print("\nglsl.Ast:\n", .{});

        try ast.print(&unbuffered_stderr.interface, allocator);

        try unbuffered_stderr.interface.print("\n", .{});
        try unbuffered_stderr.interface.flush();
    }

    var sema: glsl.Sema = .{
        .gpa = allocator,
    };
    defer sema.deinit(allocator);

    //Based on the generally correct assumption that the maximum emitted spirv instructions are roughly proportional to the number of ast nodes
    try sema.spirv_ir.node_buffer.ensureTotalCapacity(allocator, ast_node_arena_instance.queryCapacity() / 4);

    sema.spirv_ir.optimization_flags = optimization_flags;

    const errors = try sema.analyse(ast, allocator);
    defer allocator.free(errors);

    if (errors.len != 0) {
        try glsl.error_render.printErrors(ast, &sema, errors, stderr);

        return;
    }

    var schedule_context: spirv.Ir.OrderScheduleContext = .{
        .allocator = allocator,
    };
    defer schedule_context.deinit(allocator);

    try sema.spirv_ir.computeGlobalOrdering(&schedule_context, allocator);

    if (should_print_ir) {
        try stderr.print("spirv.Ir:\n\n", .{});

        try sema.spirv_ir.printNodes(stderr, schedule_context);
    }
}

test {
    std.testing.refAllDecls(@This());
}

const std = @import("std");
const spvine = @import("spvine");
const glsl = spvine.glsl;
const spirv = spvine.spirv;
const Parser = spvine.glsl.Parser;
const Ast = spvine.glsl.Ast;
const Tokenizer = spvine.glsl.Tokenizer;
