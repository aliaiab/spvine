pub fn build(builder: *std.Build) void {
    const target = builder.standardTargetOptions(.{});

    const optimize = builder.standardOptimizeOption(.{});

    _ = builder.addModule("spvine", .{
        .root_source_file = builder.path("src/main.zig"),
    });

    const exe = builder.addExecutable(.{
        .name = "spvine",
        .root_source_file = builder.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    exe.use_llvm = true;
    exe.use_lld = true;

    builder.installArtifact(exe);

    const run_cmd = builder.addRunArtifact(exe);

    run_cmd.step.dependOn(builder.getInstallStep());

    if (builder.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = builder.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const exe_tests = builder.addTest(.{
        .root_source_file = builder.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_tests = builder.addRunArtifact(exe_tests);

    const test_step = builder.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}

const std = @import("std");
