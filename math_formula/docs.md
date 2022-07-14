# Documentation for Math Formula

**WARNING:** This is still a work in progress, so not everything you see here will work in the current version.

Math Formula is an add-on that speeds up the workflow of adding nodes or node setups to a node tree in blender. A prompt can be accessed in which the user types in a "formula". This then gets converted to nodes, which can be used in blender. The syntax that can be used to write this formula is described below.

## Types

The different types are just the different types of node sockets, e.g. geometry, vector, float... 

## Syntax

**NOTE** These examples are for the Geometry Nodes editor, but most of it is applicable to shaders too.

The following example illustrates the basics of the syntax.
```js
// The type of selection will be `boolean`, because `>` returns a `boolean`.
selection = dot(normal(), {1,0,1}) > 5;
// The uv_sphere node usually also takes a radius as input, but because it was 
// omitted the node socket will be left unconnected. If we only wanted to set the
// radius and leave the rest as default we could do:
// Option 1: `uv_sphere = uv_sphere(_,_, 2.0);` or
// Option 2: `uv_sphere = uv_sphere(radius = 2.0);`
// The second option is useful if there are lots of arguments you want to skip.
uv_sphere = uv_sphere(18,20);
// Here g1 and g2 are both of the type `geometry`.
// `separate_points` refers to the `separate_geometry` node with 
g1, g2 = separate_points(uv_sphere, selection);
// Vectors can also be decomposed into their components.
// The third component is ignored in this case.
x, y = position();
// We can now update the position of g1, using the x and y components. We ignore the selection
// input.
set_position(g1,_, {x, y, z/(x*y)});
```

In essence, you have access to all the common operators like '*, +, -, /, %...'. Furthermore, the common mathematical functions like 'sin, cos, log...' are also available.

### Function overloading

Many of the common operations are possible for different types. The math formula add-on tries its best to infer types from expressions to determine the best match. This is best seen with some examples:
```js
x = 10; // x is now of type 'int'
// There is no multiplication for integers yet, so the 'Math' node set to 'Multiply' is used instead. 
y = 5 * x; // The type of y will be 'float'
// The best match is a 'Vector Math' node set to scale
z = {1,2,3} * y; // The type of z will be 'vec3'
// For 'x < y' a Compare node set to float is used.
// For 'z < 5' a Compare node set to vector is used.
// The 'or' is treated as a Boolean Math node set to 'or' is used.
test = x < y or z < 5; // The type of test will be 'boolean'.

// 'a' has not been defined, but its type can be inferred.
// The simplest possibility is for 'a' to be a 'float' to add with another 'float'.
// Hence, 'a' is seen as type 'float' and a 'Value' node is added named 'a'.
b = a + 5; 
d = c and b < a; // Similarly, 'c' will be of type 'boolean' here.
```

Remark: When the type of some variable can't be inferred, it is assumed to be a 'float'.

### Python expressions
In blender, you can type python expressions in input fields. You can also do this here by using `#`. The expression after `#` is evaluated immediately.
```js
// The difference between the two examples is the following:
// - In the first example a divide node is created, because only `tau` is evaluated as a 
//   python expression.
// - In the second example no divide node is created because `tau/4` is evaluated as a
//   python expression.
example1 = sin(#tau/4)
example2 = sin(#(tau/4))
```


### Chaining calls
When writing a formula we like to think from left to right. However, when we want to compose multiple functions we now have to think from right to left. Say that we want to scale some vector `pos` and then apply `fract` to that, we would have to write: `fract(scale(pos, 0.5))` or `fract(pos * 0.5)`. Notice that the last thing we wanted to do was the first thing we had to write. To prevent this problem you can also write the following: `pos.scale(0.5).fract();`. Calling `.function()` will take the expression before the `.` and place it in the first argument. For example: `pos.scale(0.5)` is the same as `scale(pos, 0.5)`. This can feel a lot more natural, because it is also the way we usually build node trees.

### Getting specific outputs
Another thing that can be annoying is getting the specific output of a node with many outputs. Take for example the Texture Coordinate node. If we want to get the object output you would have to write: `_,_,_, object = tex_coord();`. To fix this you can also use the `.` to get a specific output. So we could write `object = tex_coord().object`. Combining this with the chaining of function calls we get a concatenative way of writing expressions: 
```js
tex_coord().object.scale(0.5).fract().sub(0.5).abs() ...
```

### Functions Node Groups and Macros
You can define your own functions, node groups, and macros. Functions are exactly like node groups, but don't get turned into an actual node group when called. Macros on the other hand, are like an advanced "find and replace". They allow you to write commonly used expressions faster. In general if you want something with multiple nodes and arguments, using a function will be better.

### Custom Implementations
What if you want to add your own function? This is also possible with "custom implementations". In the preferences you'll find a link to the folder that contains these implementations. There are different types of files in the folder:
- Files that start with "cache": These are used by the add-on to load them faster. You should not change these.
- Files that end with "sh": These are implementations specific to shader nodes.
- Files that end with "gn": These are implementations specific to geometry nodes.
- Other files apply to both shader and geometry nodes.

What do these "implementations" look like? The general form is like this:
```rs
fn function_name(input: type, ..., input: type) -> output: type,..., output: type {
    // function body.
}
```

Let's look at an example:
```rs
// In 00_general
fn _and(a: float, b: float) -> c: float {
    // This defines the "and" operation for two floats.
    // The reason for the underscore is that `and` is a reserved keyword.
    out c = a * b;
}
```

When you create functions with names like 'add', or 'mul', the corresponding operator will also work with these implementations.
```rs
fn pow(a: vec3, b: float) -> c: float {
    ax,ay,az = a;
    out c = {ax**b, ay**b, az**b};
}

{1,2,3} ** 5; // This executes the function above
```

If you want to create a node group instead of a function, you can simply replace the `fn` with `ng`
```rs
// In 00_general
ng asinh(x: float) -> y: float {
    // The node group `asinh` takes a float "x" as input and returns a float "y".
    out y = log(x + sqrt(1+x*x), #e); // Here we set the output "y" using the `out` keyword.
}
```

**NOTE:** The node group is only created when the "function" is called.
