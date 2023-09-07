workspace "NeuralNetworks"
    configurations { "Debug", "Release" }
    includedirs { "vendor" }

project "Sandbox"
    kind "ConsoleApp"
    language "C++"
    cppdialect  "C++14"

    targetdir "%{wks.location}/bin/%{prj.name}/%{cfg.buildcfg}"
    objdir "%{wks.location}/bin/bin-int/%{prj.name}/%{cfg.buildcfg}" 
    location "%{wks.location}/build/%{prj.name}/"


    files { "%{prj.name}/**.h", "%{prj.name}/**.hpp", "%{prj.name}/**.c", "%{prj.name}/**.cpp"}
    

    filter "configurations:Debug"
        defines { "DEBUG" }
        symbols "On"

    filter "configurations:Release"
        defines { "NDEBUG" }
        optimize "On"


project "Tests"
    kind "ConsoleApp"
    language "C++"
    cppdialect  "C++14"

    links { "googletest" }

    includedirs { "googletest/googletest/include", "googletest/googletest", "Sandbox/src" }

    targetdir "%{wks.location}/bin/%{prj.name}/%{cfg.buildcfg}"
    objdir "%{wks.location}/bin/bin-int/%{prj.name}/%{cfg.buildcfg}" 
    location "%{wks.location}/build/%{prj.name}/"

    files { "%{prj.name}/**.h", "%{prj.name}/**.hpp", "%{prj.name}/**.c", "%{prj.name}/**.cpp"}


project "googletest"
    kind "StaticLib"
    includedirs { "googletest/googletest/include", "googletest/googletest" }
    files { "googletest/googletest/src/gtest-all.cc" }

    targetdir "%{wks.location}/bin/%{prj.name}/%{cfg.buildcfg}"
    objdir "%{wks.location}/bin/bin-int/%{prj.name}/%{cfg.buildcfg}" 
    location "%{wks.location}/build/%{prj.name}/"

