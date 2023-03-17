//! The analysed intermediate representation for glsl
//! Produced by IR generation from the analysed Ast produced by Sema

///Glsl source version/profile as defined by the OpenGL Language Specification
pub const GlslVersion = enum {
    unknown,

    @"110",
    @"120",
    @"130",
    @"140",
    @"150",

    @"330",

    @"410",
    @"420",
    @"430",
    @"440",
    @"450",
};

pub const GlslExtension = enum {

};

pub const Instruction = struct {
    tag: Tag,
    data: Data,

    pub const Tag = enum(u8) {
        arg,
        ret,

        constant,

        add,
        sub,
        mul,
        div,
    };

    pub const Data = union {
        arg: void,
        ret: void,
        constant: void,
        add: void,
        sub: void,
        mul: void,
        div: void,
    };
};