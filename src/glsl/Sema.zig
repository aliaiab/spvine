//! Implements the semantic analysis stage of the frontend

allocator: std.mem.Allocator,
scope_stack: std.ArrayListUnmanaged(struct {
    identifiers: token_map.Map(struct {
        type_index: u32,
    }) = .{},
}) = .{},
types: std.MultiArrayList(Type) = .{},

air_builder: struct {
    instructions: std.ArrayListUnmanaged(spirv.Air.Instruction) = .{},
    blocks: std.ArrayListUnmanaged(spirv.Air.Block) = .{},
    functions: std.ArrayListUnmanaged(spirv.Air.Function) = .{},
    variables: std.ArrayListUnmanaged(spirv.Air.Variable) = .{},
    types: std.ArrayListUnmanaged(spirv.Air.Type) = .{},
} = .{},
errors: std.ArrayListUnmanaged(Error) = .{},

pub fn deinit(
    self: *Sema,
    allocator: std.mem.Allocator,
) void {
    for (self.scope_stack.items) |*scope| {
        scope.identifiers.deinit(allocator);
    }

    self.scope_stack.deinit(allocator);
    self.types.deinit(allocator);
    self.errors.deinit(allocator);

    self.* = undefined;
}

pub const Type = struct {
    tag: Tag,
    data_start: u32,
    data_end: u32,

    pub const Tag = enum {
        literal_int,
        literal_float,
        literal_string,
        bool,
        int,
        uint,
        float,
        double,
        vec2,
        vec3,
        vec4,
        @"struct",
    };
};

pub const TypeIndex = u32;

///Analyse the root node of the ast
pub fn analyse(ast: Ast, allocator: std.mem.Allocator) !struct { spirv.Air, []Error } {
    const root_decls = ast.rootDecls();

    var sema: Sema = .{
        .allocator = allocator,
    };
    defer sema.deinit(allocator);

    try sema.scopePush();
    defer sema.scopePop();

    for (root_decls) |decl| {
        const node_tag = ast.nodes.items(.tag)[decl];

        switch (node_tag) {
            .procedure => {
                sema.analyseProcedure(ast, decl) catch |e| {
                    switch (e) {
                        error.IdentifierAlreadyDefined => {},
                        else => return e,
                    }
                };
            },
            else => {},
        }
    }

    return .{
        .{
            .capability = .Shader,
            .addressing_mode = .logical,
            .memory_model = .vulkan,
            .entry_point = .{
                .execution_mode = .vertex,
                .name = "main",
                .interface = &.{},
            },
            .instructions = &.{},
            .blocks = &.{},
            .functions = &.{},
            .variables = &.{},
            .types = &.{},
        },
        try sema.errors.toOwnedSlice(allocator),
    };
}

pub fn analyseProcedure(
    self: *Sema,
    ast: Ast,
    node: Ast.NodeIndex,
) !void {
    try self.scopePush();
    defer self.scopePop();

    const procedure = ast.dataFromNode(node, .procedure);
    const proto = ast.dataFromNode(procedure.prototype, .procedure_proto);
    const param_list = ast.dataFromNode(proto.param_list, .param_list);

    for (param_list.params) |param_node| {
        const param = ast.dataFromNode(param_node, .param_expr);

        try self.scopeDefine(ast.tokenString(param.name));
    }

    const body = ast.dataFromNode(procedure.body, .procedure_body);

    for (body.statements) |statement_node| {
        switch (ast.nodes.items(.tag)[statement_node]) {
            .statement_var_init => {
                const var_init = ast.dataFromNode(statement_node, .statement_var_init);

                try self.scopeDefine(ast.tokenString(var_init.identifier));
            },
            else => {},
        }
    }

    if (self.scopeResolve("x") != null) {
        std.log.info("Has param 'x'!!!!", .{});
    }
}

pub fn scopePush(self: *Sema) !void {
    try self.scope_stack.append(self.allocator, .{});
}

pub fn scopePop(self: *Sema) void {
    if (self.scope_stack.items.len == 0) {
        return;
    }

    const scope = &self.scope_stack.items[self.scope_stack.items.len - 1];

    scope.identifiers.deinit(self.allocator);

    self.scope_stack.items.len = 0;
}

pub fn scopeDefine(
    self: *Sema,
    identifier: []const u8,
) !void {
    const scope = &self.scope_stack.items[self.scope_stack.items.len - 1];

    if (scope.identifiers.contains(identifier)) {
        try self.errors.append(self.allocator, .{
            .identifier_redefined = .{ .identifier = identifier },
        });

        return error.IdentifierAlreadyDefined;
    }

    try scope.identifiers.put(self.allocator, identifier, .{
        .type_index = 0,
    });
}

pub fn scopeResolve(self: Sema, identifier: []const u8) ?void {
    for (1..self.scope_stack.items.len + 1) |reverse_index| {
        const index = self.scope_stack.items.len - reverse_index;
        const scope = &self.scope_stack.items[index];

        if (scope.identifiers.contains(identifier)) {
            return;
        }
    }

    return null;
}

pub const Error = union(enum) {
    identifier_redefined: struct {
        identifier: []const u8,
    },
};

const std = @import("std");
const Ast = @import("Ast.zig");
const spirv = @import("../spirv.zig");
const Sema = @This();
const token_map = @import("token_map.zig");
