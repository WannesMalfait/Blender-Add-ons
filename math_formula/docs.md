# Documentation for Math Formula

Math Formula is an addon that speeds up the workflow of adding nodes or node setups to a node tree in blender. A prompt can be accessed in which the user types in a "formula". This then gets converted to nodes, which can be used in blender. The syntax that can be used to write this formula is described below.

NOTE: These are plans, and not everything has been implemented yet.

## Types

The different types are just the different types of node sockets, e.g. geometry, vector, float... 

## Syntax

The following example illustrates the basics of the syntax.
```js
// The type of selection will be `float`, because `>` returns a `float`.
let selection = dot(normal(), {1,0,1}) > 5;
// The uv_sphere node usually also takes a radius as input, but because it was 
// omitted the node socket will be left unconnected. If we only wanted to set the
// radius and leave the rest as default we could do:
// Option 1: `uv_sphere = uv_sphere(_,_, 2.0);` or
// Option 2: `uv_sphere = uv_sphere(radius = 2.0);`
// The second option is useful if there are lots of arguments you want to skip.
let uv_sphere = uv_sphere(18,20);
// Here g1 and g2 are both of the type `Geometry`. The `separate_geometry` node has
// a mode to choose from. By default this is 'POINT', but other modes can be set by
// specifying: 
// `separate_geometry['FACE'](uv_sphere, selection);`
// Modes, or other options on the node that don't have a socket, are skipped in
// the positional argument list, i.e. you should always specify them with a keyword.
let g1, g2 = separate_geometry(uv_sphere, selection);
// Vectors can also be decomposed into their components.
let x, y = position();
// We can now update the position of g1, using the x and y components. We ignore the selection
// input.
set_position(g1,_, {x, y, z/(x*y)});
```

### Python expressions
In blender, you can type python expressions in input fields. You can also do this here by using `#`. The expression after `#` is evaluated immediately.
```js
// The difference between the two examples is the following. In the first example
// a divide node is created, because only `tau` is evaluated as a python expression.
// In the second example no divide node is created because `tau/4` is evaluated
// as a python expression.
let example1 = sin(#tau/4)
let example2 = sin(#(tau/4))
```


### Chaining calls
When writing a formula we like to think from left to right. However, when we want to compose multiple functions we now have to think from right to left. Say that we want to scale some vector `pos` and then apply `fract` to that, we would have to write: `fract(scale(pos, 0.5))` or `fract(pos * 0.5)`. Notice that the last thing we wanted to do was the first thing we had to write. To prevent this problem you can also write the following: `pos.scale(0.5).fract();`. Calling `.function()` will take the expression before the `.` and place it in the first argument. For example: `pos.scale(0.5)` is the same as `scale(pos, 0.5)`. This can feel a lot more natural, because it is also the way we usually build node trees.

### Getting specific outputs
Another thing that can be annoying is getting the specific output of a node with many outputs. Take for example the Texture Coordinate node. If we want to get the object output you would have to write: `let _,_,_, object = tex_coord();`. To fix this you can also use the `.` to get a specific output. So we could write `let object = tex_coord().object`. Combining this with the chaining of function calls we get a concatenative way of writing expressions: 
```js
tex_coord().object.scale(0.5).fract().sub(0.5).abs() ...
```

### Functions Node Groups and Macros
You can define your own functions, node groups, and macros. Functions are exactly like node groups, but don't get turned into an actual node group when called. Macros on the other hand, are like an advanced "find and replace". They allow you to write commonly used expressions faster. In general if you want something with multiple nodes and arguments, using a function will be better.


Example use of macros:
```js
// A macro definition always starts with `MACRO`. Here we define a macro
// which will replace `separate_faces` with `separate_geometry['FACE']`.
MACRO separate_faces = separate_geometry['FACE'];

// Now we can do the following:
let geo, _ = separate_faces(uv_sphere(), position().x > 0.2);
// Which is the same as:
let geo, _ = separate_geometry['FACE'](uv_sphere(), position().x > 0.2);

// NOT SUPPORTED ATM:.
MACRO lerp(a, b, fac) = map_range(fac, _,_, a, b);
MACRO slerp(a, b, fac) = map_range['SMOOTHSTEP'](fac, _,_, a, b);
MACRO sslerp(a, b, fac) = map_range['SMOOTHERSTEP'](fac, _,_, a, b);
```

NOTE: function and nodegroup creation is not supported at the moment.

As a convention you use "snake_case" for function names, and "PascalCase" for node groups. Function definition and usage:
```js
// The `fn` keyword indicates a function. Changing this to `ng` makes it a node
// node group. The inputs have names followed by their type. The outputs are
// given after the arrow (`->`). They also need to have names. To set one of
// the outputs you can access it as `self.output_name`. 
fn lin_space(a: float, b: float, num_points: int) -> points: Geometry {
    let start = {a, 0,0};
    let end = {b, 0, 0};
    self.points = mesh_line['ENDPOINTS'](count, start, end);
}

// For node groups you can specify the name of the node group, right after the 
// `ng` keyword, if it isn't provided, the function name is used.
ng "Split By Normal" SplitByNormal(geo: Geometry, vector: vector, factor: float) -> g1: Geometry, g2: Geometry {
    // Note that we use the macros defined above.
    let selection = map_range(dot(normal, vector), -1, 1, 0, 1) <= factor; 
    // Note that `g1` and `g2` are not the outputs those have to be accessed
    // with `self.`.
    let g1, g2 = separate_faces(geo, selection);
    self.g1 = g1;
    self.g2 = g2;
}

// Existing node groups can be called, even if they weren't defined using this
// add on. Here we assume a node group called "Make Wireframe" already exists.
// To call an existing node group, the name will be same except with spaces
// removed.
MakeWireframe(uv_sphere());

// If the node group has a name that consists of characters other than normal ones
// you can still call it like this:
let sum = NodeGroups["a + b"](a, b);
```
Although it's possible to define macros, node groups, and functions on the spot, they can also be placed in files to be read on startup. Node groups are never created if they aren't used.
