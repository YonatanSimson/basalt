# Local fork of basalt-headers (with EquirectangularCamera support),
# pulled in as a git submodule at thirdparty/basalt-headers-src/.
# Source: github.com/YonatanSimson/basalt-headers (fork of
# gitlab.com/VladyslavUsenko/basalt-headers).

set(SOURCE_PATH "${CURRENT_PORT_DIR}/../../../thirdparty/basalt-headers-src")

if(NOT EXISTS "${SOURCE_PATH}/CMakeLists.txt")
    message(FATAL_ERROR
        "basalt-headers submodule is empty at: ${SOURCE_PATH}\n"
        "Initialize it with:\n"
        "  git -C ${CURRENT_PORT_DIR}/../../.. submodule update --init "
        "thirdparty/basalt-headers-src"
    )
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DBUILD_TESTING=OFF
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/basalt-headers)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug" "${CURRENT_PACKAGES_DIR}/lib")

file(INSTALL "${CURRENT_PORT_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
