# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.core.plugins import Plugins
from hydra.plugins.plugin import Plugin

from pathlib import Path
LOCAL_PATH = str(Path(__file__).parents[1].resolve())


class ExampleSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Appends the search path for this plugin to the end of the search path
        # Note that foobar/conf is outside of the example plugin module.
        # There is no requirement for it to be packaged with the plugin, it just needs
        # be available in a package.
        # Remember to verify the config is packaged properly (build sdist and look inside,
        # and verify MANIFEST.in is correct).
        search_path.append(
            provider="example-searchpath-plugin", path=f"{LOCAL_PATH}"
        )
        
Plugins.instance().register(ExampleSearchPathPlugin)