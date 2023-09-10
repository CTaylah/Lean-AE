workspace "NeuralNetworks"
    configurations { "Debug", "Release", "DebugTest" }
    includedirs { "vendor" }
    filter "configurations:DebugTest"
        defines { "DEBUG_TEST" }

    
project "NeuralNetwork"
    language "C++"
    cppdialect  "C++14"
    kind "StaticLib"

    targetdir "%{wks.location}/bin/%{prj.name}/%{cfg.buildcfg}"
    objdir "%{wks.location}/bin/bin-int/%{prj.name}/%{cfg.buildcfg}" 
    location "%{wks.location}/build/%{prj.name}/"

    includedirs { "%{prj.name}/include" }

    files {"%{prj.name}/**.hpp", "%{prj.name}/**.c", "%{prj.name}/**.cpp"}

    filter "configurations:Debug"
        defines { "DEBUG" }
        symbols "On"

    filter "configurations:Release"
        defines { "RELEASE" }
        optimize "On"
    


    

project "Tests"
    kind "ConsoleApp"
    language "C++"
    cppdialect  "C++14"

    links { "googletest" }
    links { "NeuralNetwork" }

    includedirs { "%{wks.location}/googletest/googletest/include", "%{wks.location}/googletest/googletest", "%{wks.location}/NeuralNetwork/include" }

    targetdir "%{wks.location}/bin/%{prj.name}/%{cfg.buildcfg}"
    objdir "%{wks.location}/bin/bin-int/%{prj.name}/%{cfg.buildcfg}" 
    location "%{wks.location}/build/%{prj.name}/"

    files {"%{prj.name}/**.hpp", "%{prj.name}/**.c", "%{prj.name}/**.cpp"}

--Third Party
project "googletest"
    kind "StaticLib"
    includedirs { "googletest/googletest/include", "googletest/googletest" }
    files { "googletest/googletest/src/gtest-all.cc" }

    targetdir "%{wks.location}/bin/%{prj.name}/%{cfg.buildcfg}"
    objdir "%{wks.location}/bin/bin-int/%{prj.name}/%{cfg.buildcfg}" 
    location "%{wks.location}/build/%{prj.name}/"

