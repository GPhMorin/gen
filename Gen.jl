using DataStructures, Memoize

struct Genealogy
    filename::String
    parents::DefaultDict{Int32, Tuple{Int32, Int32}}
    index::Dict{Int32, Int32}

    function Genealogy(filename::String)
        lines = open(filename, "r") do f
            readlines(f)[2:end]
        end
        parents, index = load_dictionaries(lines)
        new(filename, dict)
    end
end

function load_dictionaries(lines::Vector{String})
    idx = 1
    parents = DefaultDict{Int32, Tuple{Int32, Int32}}((0, 0))
    index = Dict{Int32, Int32}()
    for line in lines
        child, father, mother = split(strip(line), '\t')
        father = parse(Int32, father)
        mother = parse(Int32, mother)
        if father != 0 || mother != 0
            parents[parse(Int32, child)] = (father, mother)
        end
        index[parse(Int32, child)] = idx
        idx += 1
    end
    parents, index
end

function probands(dict::DefaultDict{Int32, Tuple{Int32, Int32}})
    individuals = keys(dict)
    fathers = Set([dict[individual][1] for individual in individuals])
    mothers = Set([dict[individual][2] for individual in individuals])
    parents = union(fathers, mothers)
    probands = [individual for individual ∈ individuals if individual ∉ parents]
    return probands
end

function unique_family_members(dict::DefaultDict{Int32, Tuple{Int32, Int32}})
    pros = probands(dict)
    visited_parents = Set{Tuple{Int32, Int32}}()
    unique_members = Vector{Int32}()
    for pro in pros
        father, mother, index = dict[pro]
        if (father, mother) ∉ visited_parents
            push!(unique_members, pro)
            push!(visited_parents, (father, mother))
        end
    end
    return unique_members
end

@memoize function kinship(parents::DefaultDict{Int32, Tuple{Int32, Int32}}, index::Dict{Int32, Int32}, individual1::Int32, individual2::Int32)
    if individual1 == individual2
        father, mother = parents[individual1]
        value = 0.0
        if father != 0 && mother != 0
            value = kinship(parents, index, father, mother)
        end
        return (1 + value) * 0.5
    end

    if index[individual2] > index[individual1]
        individual1, individual2 = individual2, individual1
    end

    father, mother = parents[individual1]
    if father == 0 && mother == 0
        return 0.0
    end

    mother_value = 0.0
    father_value = 0.0
    if mother != 0
        mother_value = kinship(parents, index, mother, individual2)
    end
    if father != 0
        father_value = kinship(parents, index, father, individual2)
    end
    return (father_value + mother_value) / 2.0
end