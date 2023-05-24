using DataStructures, Memoize

struct Genealogy
    filename::String
    dict::Dict{Int32, Tuple{Union{Int32, Nothing}, Union{Int32, Nothing}, Int32}}
end

function Genealogy(filename::String)
    lines = open(filename, "r") do f
        readlines(f)[2:end]
    end
    dict = load_dictionary(lines)
    Genealogy(filename, dict)
end

function load_parents(lines::Vector{String})
    parents = OrderedDict{Int32, Tuple{Union{Int32, Nothing}, Union{Int32, Nothing}}}()
    for line in lines
        # Splitting the line with multiple possible separators
        child, father, mother = split(strip(line), '\t')
        father = father != "0" ? parse(Int, father) : nothing
        mother = mother != "0" ? parse(Int, mother) : nothing
        parents[parse(Int, child)] = (father, mother)
    end
    parents
end

function load_dictionary(lines::Vector{String})
    index = 1
    dict = Dict{Int, Tuple{Union{Int32, Nothing}, Union{Int32, Nothing}, Int32}}()
    for line in lines
        child, father, mother = split(strip(line), '\t')
        father = father != "0" ? parse(Int32, father) : nothing
        mother = mother != "0" ? parse(Int32, mother) : nothing
        dict[parse(Int32, child)] = (father, mother, index)
        index += 1
    end
    dict
end


function ancestors(g::Genealogy, descendant::Int32)
    ancestors = Set{Int32}()
    stack = Vector{Int32}()
    push!(stack, descendant)
    while !isempty(stack)
        current = pop!(stack)
        father, mother = g.parents[current]
        if father != nothing && father ∉ ancestors
            push!(stack, father)
            push!(ancestors, father)
        end
        if mother != nothing && mother ∉ ancestors
            push!(stack, mother)
            push!(ancestors, mother)
        end
    end
    ancestors
end

function common_ancestors(g::Genealogy, individual1::Int32, individual2::Int32)
    common_ancestors = Set{Int32}()
    push!(common_ancestors, individual1)
    push!(common_ancestors, individual2)
    ancestors1 = ancestors(g, individual1)
    ancestors2 = ancestors(g, individual2)
    intersection = intersect(ancestors1, ancestors2)
    union!(common_ancestors, intersection)
    common_ancestors
end

function probands(g::Genealogy)
    individuals = keys(g.dict)
    fathers = Set([g.dict[individual][1] for individual in individuals])
    mothers = Set([g.dict[individual][2] for individual in individuals])
    parents = union(fathers, mothers)
    probands = [individual for individual ∈ individuals if individual ∉ parents]
    probands
end

function bfs(g::Genealogy, start::Int32, goal::Int32)
    queue = Deque{Tuple{Int32, Vector{Int32}}}()
    push!(queue, (start, [start]))
    paths = Vector{Vector{Int32}}()

    while !isempty(queue)
        node, path = popfirst!(queue)

        if node == goal
            push!(paths, path)
        else
            neighbors = g.parents[node]
            for neighbor in neighbors
                if neighbor != nothing && neighbor ∉ path
                    new_path = vcat(path, [neighbor])
                    push!(queue, (neighbor, new_path))
                end
            end
        end
    end
    paths
end

function inbreeding(g::Genealogy, individual::Int32)
    father, mother = g.parents[individual]
    if father == nothing && mother == nothing
        return 0.0
    end

    if father == mother
        return 0.5 * (1.0 + inbreeding(g, father))
    end

    ancestors = common_ancestors(g, father, mother)
    if isempty(ancestors)
        return 0.0
    end

    F = 0.0

    for common_ancestor in ancestors
        Fca = inbreeding(g, common_ancestor)
        paths1 = bfs(g, father, common_ancestor)
        paths2 = bfs(g, mother, common_ancestor)
        possible_loops = Vector{Vector{Int32}}()
        for path1 in paths1
            for path2 in paths2
                if length(path1) - length(Set(path1)) == 1
                    push!(possible_loops, path1 + path2)
                end
            end
        end
        for loop in possible_loops
            F += 0.5 ^ length(Set(loop)) * (1.0 + Fca)
        end
    end

    F
end

function iterative_kinship(g::Genealogy, individual1::Int32, individual2::Int32)
    ϕ = 0.0

    if individual1 == individual2
        father, mother = g.parents[individual1]
        ϕ = 0.5 * (1.0 + iterative_kinship(g, father, mother))
        ϕ
    else
        ancestors = common_ancestors(g, individual1, individual2)
        if !isempty(ancestors)
            for ancestor in ancestors
                paths1 = bfs(g, individual1, ancestor)
                paths2 = bfs(g, individual2, ancestor)
                possible_loops = [vcat(path1, path2) for path1 in paths1, path2 in paths2]
                loops = [loop for loop in possible_loops if length(loop) - length(Set(loop)) == 1]
                for loop in loops
                    ϕ += 0.5 ^ length(Set(loop))
                end
            end
        end

        ϕ
    end
end

@memoize function recursive_kinship(dict::Dict, individual1::Int32, individual2::Int32)
    if individual1 == individual2
        father, mother, _ = dict[individual1]
        if father != nothing && mother != nothing
            value = recursive_kinship(dict, father, mother)
        else
            value = 0.0
        end
        return (1 + value) * 0.5
    end

    if dict[individual2][3] > dict[individual1][3]
        individual1, individual2 = individual2, individual1
    end

    father, mother, _ = dict[individual1]
    if father == nothing && mother == nothing
        return 0.0
    end

    mother_value = 0.0
    father_value = 0.0
    if mother != nothing
        mother_value = recursive_kinship(dict, mother, individual2)
    end
    if father != nothing
        father_value = recursive_kinship(dict, father, individual2)
    end
    return (father_value + mother_value) / 2.0
end
