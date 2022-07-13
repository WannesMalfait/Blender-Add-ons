import pickle
import os
import bpy
from math_formula.backends.type_defs import TyFunction
from math_formula.compiler import Compiler
from math_formula.parser import Error


class FileData():
    def __init__(self) -> None:
        self.geometry_nodes: dict[str, list[TyFunction]] = {}
        self.shader_nodes: dict[str, list[TyFunction]] = {}

    def num_funcs(self) -> int:
        tot = 0
        for value in self.geometry_nodes.values():
            tot += len(value)
        for value in self.shader_nodes.values():
            tot += len(value)
        return tot


add_on_dir = os.path.dirname(
    os.path.realpath(__file__))
custom_implementations_dir = os.path.join(add_on_dir, 'custom_implementations')


@bpy.app.handlers.persistent
def load_custom_implementations(dir: str = None, force_update: bool = False) -> list[tuple[str, list[Error]]]:
    if dir is None:
        prefs = bpy.context.preferences.addons['math_formula'].preferences
        dir = prefs.custom_implementations_folder
    filenames = os.listdir(dir)
    # Ensure that we load files in the right order.
    filenames.sort()
    errors = []
    file_data.geometry_nodes = {}
    file_data.shader_nodes = {}
    if not force_update:
        for filename in filenames:
            if filename.startswith('cache'):
                with open(os.path.join(dir, filename), 'rb') as f:
                    cached = pickle.load(f)
                    if filename.endswith('_gn'):
                        file_data.geometry_nodes = cached
                    elif filename.endswith('_sh'):
                        file_data.shader_nodes = cached
                    f.close()
        return errors
    geo_compiler = Compiler('GeometryNodeTree')
    sha_compiler = Compiler('ShaderNodeTree')
    for filename in filenames:
        if not filename.startswith('cache'):
            with open(os.path.join(dir, filename), 'r') as f:
                source = f.read()
                if filename.endswith('_gn') or not filename.endswith('_sh'):
                    succeeded = geo_compiler.check_functions(source)
                    if not succeeded:
                        errors.append((filename, geo_compiler.errors))
                if filename.endswith('_sh') or not filename.endswith('_gn'):
                    succeeded = sha_compiler.check_functions(source)
                    if not succeeded:
                        errors.append((filename, sha_compiler.errors))
                f.close()
    if errors != []:
        return errors
    # Store everything if succesful.
    file_data.geometry_nodes = geo_compiler.type_checker.functions
    file_data.shader_nodes = sha_compiler.type_checker.functions
    # Add to cache
    with open(os.path.join(dir, 'cache_gn'), 'w+b') as f:
        pickle.dump(file_data.geometry_nodes, f)
        f.close()
    with open(os.path.join(dir, 'cache_sh'), 'w+b') as f:
        pickle.dump(file_data.shader_nodes, f)
        f.close()
    return errors


class MF_OT_load_custom_implementations(bpy.types.Operator):
    """Load all the custom implementations from the given folder.
If there is a cache it will be loaded from there instead,
unless `force_update` is true"""
    bl_idname = "node.mf_load_custom_implementations"
    bl_label = "Load custom implementations"
    bl_options = {'REGISTER', 'UNDO'}

    force_update: bpy.props.BoolProperty(
        name="Force Update",
        description="Force reparsing of files, even if there is a cache file",
        default=False
    )

    def execute(self, context: bpy.context):
        prefs = context.preferences.addons['math_formula'].preferences
        errors = load_custom_implementations(
            prefs.custom_implementations_folder, self.force_update)
        if errors != []:
            self.report(
                {'ERROR'}, 'Errors when loading macros. See console for more details.')
            for filename, file_errors in errors:
                print(f'Errors in {filename}')
                for file_error in file_errors:
                    print(file_error)
            # Ensure that we don't use wrong implementations
            file_data.geometry_nodes = {}
            file_data.shader_nodes = {}
            return {'CANCELLED'}
        self.report(
            {'INFO'}, f'Succesfully loaded {file_data.num_funcs()} custom implementations.')
        return {'FINISHED'}


classes = (
    MF_OT_load_custom_implementations,
)


def register():
    global file_data
    file_data = FileData()
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.app.handlers.load_post.append(load_custom_implementations)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
