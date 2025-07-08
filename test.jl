"""
	function_name(arg1::String, arg2::String) -> String

Brief description of the function.

# Arguments
- `arg1::String`: Description of arg1
- `arg2::String`: Description of arg2

# Returns
- `String`: Description of return value

# Examples
```julia
result = function_name(1, "hello")
"""
function function_name(arg1::String, arg2::String)::String
	return "Hello, World!"
end

a = function_name(2, "hello")
