# Blender-Add-ons
This contains multiple free to use add-ons for blender.
Just open one of the files and download it.
You can also copy-paste the code into the blender text-editor to test out the script there.
Feel free to report issues or to make pull requests.

## L-system
Generate a fractal structure based on Lindenmayer systems. See https://en.wikipedia.org/wiki/L-system for some examples.

## Supershape
Generate a 3D model based on the "Superformula", useful to create some abstract objects.

## Times Table
Generate a nice pattern. Adapted from Mathologer's video: https://www.youtube.com/watch?v=qhbuKbxJsk8.

## View Finder
A helper add-on to speed up the process of finding the best values for your current node setup.
Current features:
- Generate a 2D or 3D scene for comparing values.
- Generate a sequence of images for different values of a node input socket.

## Math Formula

An add-on to speed up adding math nodes to your node tree. You can access the menu with `SHIFT + F` (with attributes) or `ALT +F` (normal math nodes).
### General syntax
This is the syntax with which you can build your formulas:
- `(` and `)` can be used to give precedence in the order of operations, e.g `(1 + 2) * 4` evaluates to `12` and not `9`.
- Common operators which have a special symbol can be placed in between their operands. These operators are: `> < + - * / % ^ **`, These also have a priority level.  The priorities are as follows: `function call` > `( )` > `** ^` > `* / %` > `+ -` > `< >`
- The same operators can also be used as vector math operations by prefixing it with a `v`, if the respective vector math operation is part of the vector math node. So `*` becomes `v*`, for example.
- All other functions have to be called in the normal way present in most programming languages: `function_name(arg1, arg2, arg3)`, with the arguments separated by comma's. It is possible that the same function has multiple aliases, e.g. `sine` and `sin` are the same function. The arguments can be expressions themselves., e.g. `sin(4 * 5)` is valid syntax.
- Creating a vector is done with `{ }`, with the components separated by comma's. So the vector with components `4` `5` and `7` would be written as `{4, 5, 7}`. If a vector contains inputs that aren't floats, a `Combine XYZ` node is added instead. This means that `{4, 5, 7*8}` will be a `Combine XYZ` node, because the final component is an expression.
- In blender, it is possible to type in mathematical equations in the float fields. The special function `!()`, evaluates the expression inside the parentheses with python. The expression must be a python expression that evaluates to a number. For convenience, the python `math` library will be imported so common constants and mathematical functions are available. Example: `sin(!(pi/4)*x)`.
- The semicolon, `;`, marks the end of every statement. Although it is not strictly necessary, it's recommended to place it in between statements, so that errors can be displayed correctly.
- If a function takes in a vector as an argument, but a float is given, then it is automatically converted to a vector. So `vsin(10)` is treated as `vsin({10,10,10})`.
- `=` can be used to assign/declare variables. See below for specific rules. It is also an expression, i.e. `sin(x=2)` is the same as `x=2; sin(x)`.

### Syntax specific to normal math nodes:
- The `let` keyword can be used to declare variables without specifying a value. For example `let x;`, is the same as `x = 2` or `let x = 2`. Declaring a variable is the same as adding an input node, with the label set to the variable name. Later uses of that variable name will connect the output of that input node to the appropriate input socket. 
- To create a `Separate XYZ` node with specific variable names you can use the syntax `let {name_1,name_2,name_3} = some_name;`. The right hand side can be an expression. Otherwise `.` followed by any of `x`, `y` or `z`, will create a `Separate XYZ` node from the result of the left hand side. Example: `vfract(scale(coords, 10)).xy`.
- All other kinds of variables can be assigned a value without needing to use `let`.  Example: `z = sin(x);`, will store the output socket of the right hand side in the variable `z`, which can then be used later.
- Undeclared variables will be declared automatically when used. For example, the input `let {a,b,c}= var_name;` would be treated as `var_name = 0; let {a,b,c} = var_name`.

### Syntax specific to attribute math nodes:
- Unlike with normal math nodes, in this case the leftover words will be treated as input attributes.
- The operator `=` can be used to set the name of the next attribute math operation. For example: `z = sin(x);` would create an attribute math node with the operation set to `SINE`, the input set to the attribute `x`, and the result set to the attribute `z`. So, continuing on the previous example, if we now wanted to multiply `z` with `x` we'd have to do this: `z = sin(x); z*x;` or `(z=sin(x))*x`.
- Appending `.` followed by any of `x`, `y` or `z` to an attribute, will create a `Separate XYZ` node with the input set to the attribute and the resulting components set if they appeared after the `.`. For example `position.yz` creates a `Separate XYZ` node with the input attribute set to `position`, and the outputs: `Result Y` and `Result Z` set to `y` and `z` respectively. The attribute output, `Result X`, is left empty.
- Like with normal math nodes, `let {x_name, y_name, z_name}` followed by a vector of three names  can be used to create a `Separate XYZ` node. The three components of the vector are treated as attribute names, and not as expressions to be evaluated. Example: `let { ,pos_y, pos_z} = position;`, creates attributes `pos_y` and `pos_z`, set to the `y` and `z` components of the attribute `position`.

### Examples:
- `position.xy; position = {x, y, x**2+y**2};`
- `position.xy; z = x^2+y^2; position = {x, y, z};`
- `vfract(vscale(coords, 10)) v- 0.5;`
- `vabs(coords v- 0.5).xy; smax(x,y,0.2) < 0.2`
- `coords.xyz; r = length({x,y,0}); theta = atan2(y,x); {r,theta,z};`




