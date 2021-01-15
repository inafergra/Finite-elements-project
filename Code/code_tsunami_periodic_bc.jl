using Pkg
Pkg.activate(".")
Pkg.instantiate()

using CompScienceMeshes
using LinearAlgebra
using SparseArrays

fn = joinpath(@__DIR__, "assets", "world.msh")
border = CompScienceMeshes.read_gmsh_mesh(fn, physical="Border", dimension=1)
coast  = CompScienceMeshes.read_gmsh_mesh(fn, physical="Coast", dimension=1)
sea    = CompScienceMeshes.read_gmsh_mesh(fn, physical="Sea", dimension=2)

# skeleton creates lower dimensional meshes from a given mesh. With second argument
# zero the returned mesh is simply the cloud of vertices on which the original mesh
# was built.

border_vertices = skeleton(border, 0)
coast_vertices  = skeleton(coast, 0)
sea_vertices    = skeleton(sea, 0)

#When solving the equation div(grad(u))-k^2*u=f the active vertices (i.e., the vertices with basis functions associated) only exclude the vertices in the coast line(\Gamma_{1}); border vertices do have basis functions attached

active_vertices = submesh(sea_vertices) do v 
    #v in border_vertices && return false   
    v in coast_vertices && return false #we exclude the coast vertices
    return true
end

"""
The function 'localtoglobal'creates the local to global map for FEM assembly.

    localtoglobal(active_vertices, domain) -> gl

The returned map `gl` can be called as

    gl(k,p)

Here, `k` is an index into `domain` (i.e. it refers to a specific element, and
`p` is a local index into a specific element. It ranges from 1 to 3 for triangular
elements and from 1 to 2 for segments. The function returns an index `i` into
`active_vertices` if the i-th active vertex equals the p-th vertex of element k and
`gl` return `nothing` otherwise.
"""

function localtoglobal(active_vertices, domain)
    conn = copy(transpose(connectivity(active_vertices, domain, abs)))
    nz = nonzeros(conn)
    rv = rowvals(conn)
    function gl(k,p)
        for q in nzrange(conn,k)
            nz[q] == p && return rv[q]
        end
        return nothing
    end
    return gl
end

#-------------------------------------------------------------------------------

#this function finds the indices and coordinates of the vertices in the sides (border) 

function get_side_indices(active_vertices, mesh)

    gl = localtoglobal(active_vertices, mesh)
    
    #dictionaries to save the coordinates and the indices of the sides
    top = Dict{String, Array}("index"=>[], "coords"=>[])
    bottom = Dict{String, Array}("index"=>[], "coords"=>[])
    left = Dict{String, Array}("index"=>[], "coords"=>[])
    right = Dict{String, Array}("index"=>[], "coords"=>[])
    for (k, element) in enumerate(mesh)
        for p in 1:length(element)
            (x, y, z) = border.vertices[element[p]]
            if x == -1
                push!(left["index"], gl(k,p))
                push!(left["coords"], y)
            end
            if x == 1
                push!(right["index"], gl(k,p))
                push!(right["coords"], y)
            end
            if y == -1
                push!(bottom["index"], gl(k,p))
                push!(bottom["coords"], x)
            end
            if y == 1
                push!(top["index"], gl(k,p))
                push!(top["coords"], x)
            end
        end
    end

    sides = Dict{String, Array}(
        "left" => unique(left["index"][sortperm(left["coords"])]),
        "right" => unique(right["index"][sortperm(right["coords"])]),
        "top" => unique(top["index"][sortperm(top["coords"])]),
        "bottom" => unique(bottom["index"][sortperm(bottom["coords"])])
       )

    return sides
end

#new local to global function assigning the same index to the points with same y (x) in the border left and right (top and bottom) border. These pair of points need to have the same basis
#functions associated, meaning that the global index gl should be the same.
function localtoglobal_periodic(active_vertices, domain, sides)
    gl = localtoglobal(active_vertices, domain)
    function gl_new(k,p)
        index = gl(k,p)
        index in sides["bottom"] && return sides["top"][findall(x->x==index, sides["bottom"])][1]
        index in sides["right"] && return sides["left"][findall(x->x==index, sides["right"])][1]
        return index
    end
    return gl_new
end

#------------------------------------------------------------------------------------

#internal element matrix
function elementmatrix(mesh, element, kconstant)
    v1 = mesh.vertices[element[1]]
    v2 = mesh.vertices[element[2]]
    v3 = mesh.vertices[element[3]]
    tangent1 = v3 - v2
    tangent2 = v1 - v3
    tangent3 = v2 - v1
    normal = (v1-v3) × (v2-v3)
    area = 0.5 * norm(normal)
    normal = normalize(normal)
    grad1 = (normal × tangent1) / (2 *area)
    grad2 = (normal × tangent2) / (2 *area)
    grad3 = (normal × tangent3) / (2 *area)
    S = area *( [
        dot(grad1,grad1) dot(grad1,grad2) dot(grad1,grad3)
        dot(grad2,grad1) dot(grad2,grad2) dot(grad2,grad3)
        dot(grad3,grad1) dot(grad3,grad2) dot(grad3,grad3)] - (Matrix(I, 3,3) + ones(3,3)) * kconstant^2 / 3)

    return S
end

#assembly function for the internal matrix
function assemblematrix(mesh, active_vertices, kconstant, sides)
    n = length(active_vertices)
    S = zeros(ComplexF64,n,n)
    gl = localtoglobal_periodic(active_vertices, mesh, sides)
    for (k,element) in enumerate(mesh)
        Sel = elementmatrix(mesh, element, kconstant)
        for p in 1:3
            i = gl(k,p) #i is the global index corresponding to the local index p of element k 
            i == nothing && continue
            for q in 1:3
                j = gl(k,q)
                j == nothing && continue
                S[i,j] += Sel[p,q]
            end
        end
    end
    return S
end

#------------------------------------------------------------------------------------

#element vector
function elementvector(f, mesh, element)
    v1 = mesh.vertices[element[1]]
    v2 = mesh.vertices[element[2]]
    v3 = mesh.vertices[element[3]]
    el_size = norm((v1-v3)×(v2-v3))/2 #area
    F = el_size * [
        f(v1)/3
        f(v2)/3
        f(v3)/3]
    return F
end

#assembly function for the element vector
function assemblevector(f, mesh, active_vertices, sides)
    n = length(active_vertices)
    F = zeros(ComplexF64,n)
    gl =  localtoglobal_periodic(active_vertices, mesh, sides)
    for (k,element) in enumerate(mesh)
        Fel = elementvector(f,mesh,element)
        for p in 1:3
            i = gl(k,p)
            i == nothing && continue
            F[i] += Fel[p]
        end
    end
    return F
end


#-----------------------------------------------------------------------------------

kconstant = 6*pi
sides = get_side_indices(active_vertices, sea)

#assembly of the internal matrix
S = assemblematrix(sea, active_vertices, kconstant, sides)

#assembly of the element vector
function f(x) #gaussian function
    f = exp(-100.0*(x[1]+1.5)^2) * exp(-100.0*(x[2]-1.5)^2 )
end

b = assemblevector(f, sea, active_vertices, sides)


eff_indices = [] #eliminating the vertices from the bottom and right borders
for i in 1:length(active_vertices)
    i in sides["bottom"] && continue
    i in sides["right"] && continue
    push!(eff_indices, i)
end

S = S[eff_indices, eff_indices]
b = b[eff_indices]

#solving the linear system of equation S·c=b
u_eff = S \ b

#plotting
u = zeros(ComplexF64,length(sea_vertices)) #solu
for (i, index) in enumerate(eff_indices)
    u[index] = u_eff[i]
end

#now we add the solution for the bottom and right border (which is the same for the top and left borders, respectively)
for i in 1:length(sides["bottom"])
    u[sides["bottom"][i]] = u[sides["top"][i]]
end
for i in 1:length(sides["right"])
    u[sides["right"][i]] = u[sides["left"][i]]
end

u_tilda = zeros(ComplexF64,length(sea_vertices))
for (j,m) in enumerate(active_vertices)
    u_tilda[m[1]] = u[j]
end

using Makie
# scene = Makie.mesh(vertexarray(sea), cellarray(sea), color=real(u_tilda))

p = Makie.mesh(vertexarray(sea), cellarray(sea), color=real(u_tilda))
cm = colorlegend(
    p1[end],             
    raw = true,          
    camera = campixel!,  
                            
    width = (           
        30,              
        560              
    )
    )

scene_final = vbox(p, cm)

Makie.save("finemesh0_periodic.png", scene_final)