project "Sandbox"
    kind "ConsoleApp"
    language "C++"
    cppdialect  "C++14"

    targetdir "%{wks.location}/bin/%{prj.name}/%{cfg.buildcfg}"
    objdir "%{wks.location}/bin-int/%{prj.name}/%{cfg.buildcfg}" 
    location "%{wks.location}/build/%{prj.name}/"

    includedirs 
    {
        "%{wks.location}/vendor/"
    }

    files { "**.h", "**.hpp", "**.c", "**.cpp"}
    
    filter "configurations:Debug"
        defines { "DEBUG" }
        symbols "On"
    
    filter "configurations:Release"
        defines { "NDEBUG" }
        optimize "On"


