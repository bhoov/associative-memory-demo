### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 70d5be76-0a5d-11eb-2626-6799926deaeb
begin
    import Pkg
    Pkg.activate(mktempdir())
	Pkg.add([
		"Images",
		"ImageMagick",
		"Compose",
		"ImageFiltering",
		"TestImages",
		"Statistics",
		"PlutoUI",
		"Memoize",
		"BenchmarkTools",
		"MLDatasets",
		"Colors",
		"StatsBase",
		"DelimitedFiles",
		"JSON2",
		"Tables",
		"PyCall",
		"AbstractTrees",
		"Conda"
		])
	
	using Tables
	using PyCall
	using Conda
	using JSON2
	using DelimitedFiles
    using Images
	using StatsBase
    using PlutoUI
	using MLDatasets
	using Memoize
	using AbstractTrees
	import Base
	
	md"> Creating the environment"
end

# ╔═╡ b105d9c2-0a79-11eb-2f6f-4b12f9498f89
md"""
# Playing with Continuous Modern Hopfield Networks

Math and inspiration taken from the [Hopfield Networks is All You Need Blog](https://ml-jku.github.io/hopfield-layers/)
"""

# ╔═╡ 08ec4722-6761-11eb-1b5c-d598cc7abaea
md"""
## Helper Code
"""

# ╔═╡ e37dcd7e-0a79-11eb-1273-2f35d70a42a1
md"""
## Hopfield Layer Examples

### Finding a Memory

- Select a **seed index** to select any memory the model has seen (unless you are testing for unseen memories, see below).
- Modify the total **number of memories** stored by the model (from 1 -> reasonable max number). /Will perform slower at higher numbers/
- Control how much to **obscure** the state vector.
- Change the **inverse temperature** parameter β
"""

# ╔═╡ ed96a526-0a68-11eb-0dd8-33787eab6779
begin
	NumberOfMemories = @bind nMemories Slider(5:100:30000, show_value=true, default=30)
	md"""**NumberOfMemories** = $(NumberOfMemories)"""
end

# ╔═╡ b2ff9d5a-0a6d-11eb-0803-792ae4dd0997
begin
	PercentObscured = @bind pctAffected Slider(0:0.02:1.0, default=0.5, show_value=true)
	md"""**PercentObscured** = $(PercentObscured)"""
end

# ╔═╡ 7de67790-0a6f-11eb-2b9f-91ec6a0035e9

begin
	Beta = @bind β Slider(0.001:0.1:800, show_value=true)
	md"""**Beta** = $(Beta)"""
end


# ╔═╡ 82f2fbe0-0a7a-11eb-2faf-5730af229577
md"""
Seed displayed on the left. Retrieved image on the right
"""

# ╔═╡ bad039e0-69ba-11eb-0f3b-4d4e747f9142
md"""
You can see how the distribution of the classification pattern changes:
"""

# ╔═╡ 9d4ffdc6-0a7a-11eb-3e52-4f7c39e6f8d2
md"""
### Now what if we search for a pattern that was never stored?

Look what happens if we change the seed index to only include memories not seen by the model
"""

# ╔═╡ dea5fafa-0a7a-11eb-2db7-a16b04aa4b65
begin
	ShowUnseen = @bind showUnseen CheckBox(default=false) 
	md"""**ShowUnseen** = $(ShowUnseen)"""
end

# ╔═╡ a74409a2-0a7b-11eb-2b15-b51ae49b13fb
md"""
It looks like the model develops a really incomplete understanding of the pattern even if there are patterns that are very similar
"""

# ╔═╡ b06e8c6c-6653-11eb-1e2c-6948e68e20fb
md"""
### Simple Multimodal example

Hopfield networks can easily handle data in mutliple domains, so long as that data can be represented as a vector. This is done by concatenating the modality vectors of each instance.

In an overly simplified world, if we wanted to associate text and images, we could create a label like 0-9 for each of the MNIST examples and append them to each MNIST vector
"""

# ╔═╡ e235e2b6-668c-11eb-2cec-879584be1e27
begin
	WithLabel = @bind withLabel CheckBox(default=false)
	md"""**Treat as Classification Problem** = $(WithLabel)"""
end


# ╔═╡ abf2e53a-6652-11eb-1cbf-8d0cef72a087
md"""
### More than MNIST

This works for other image domains too, like CIFAR
"""

# ╔═╡ 715eb624-6652-11eb-337a-4b6fe9950763

begin
	UseCifar = @bind useCifar CheckBox(default=false)
	md"""Use CIFAR = $(UseCifar)"""
end


# ╔═╡ df4e3e08-0a5d-11eb-331c-59c2cd60bb4a
begin
	function loadMnist()
		x, y = MNIST.traindata();
		x = Float64.(permutedims(x, (2,1,3)))
		
		allKeys = sort(unique(y))
		allVals = collect(0:9)
		nLabels = 10
		
		"Labels always range 1:nLabels. Index into correct string"
		function getClassname(label)
			x2ind(i) = Int(round(clamp(i, 1, nLabels)))
			return allVals[x2ind(label)]
		end
		
		function getLabel(classname)
			results = findall(x -> x == classname, allVals)
			@assert length(results) > 0 "Could not find $(classname) in available values"
			return results[1]
		end

		return x, y.+1, getClassname, getLabel
	end
	
	function loadCifar()
		x, y = CIFAR10.traindata();
		fix_order = permutedims(x, (3,2,1,4))
		grayImg = dropdims(mean(fix_order, dims=1), dims=1)
		x = Float64.(grayImg)
		# x = colorview(Gray, grayImg)
		allKeys = sort(unique(y))
		allVals = CIFAR10.classnames()
		nLabels = length(allVals)
		
		function getClassname(label)
			allVals[Int(label)]
		end
		
		function getLabel(classname)
			results = findall(x -> x == classname, allVals)
			@assert length(results) > 0 "Could not find $(classname) in available values"
			return results[1]
		end
		
		return x, y.+1, getClassname, getLabel
	end

	train_x, train_y, getClassname, getLabel = useCifar ? loadCifar() : loadMnist()
	x, y, n_examples = size(train_x)
	
	md"> Data Loading Functions"
end

# ╔═╡ f08135ee-0a77-11eb-2caa-ab85b93ddb38
begin
	@memoize function cache_sample(a, wv)
		return sample(a, wv, replace=false)
    end
	
	"""
	Create a random index order up to length n
	"""
	@memoize function randomIdxOrder(n::Int)
		return sample(1:n, n, replace=false)
	end
	
	"""
	Create a random index order up to length n
	"""
	@memoize Dict function randomIdxOrder(a::Array{Any})
		return randomIdxOrder(length(a))
	end
	
	"""Convert single MNIST image into flat memory"""
	function flattenImg(img::Array)
		return reshape(img, x*y)
	end
	
	"""View a memory as an image"""
	function viewMem(arr::Array)
		reshape(arr, x, y)
	end
	
	"""Obscure an array for an incomplete pattern"""
	function obscureMem(arr::Array, fracAffected=0.9)
		newState = copy(arr)
		n = length(arr)
		nAffected = floor(Int, fracAffected * n)
		affectedIdxs = randomIdxOrder(n)[1:nAffected]
		# affectedIdxs = cache_sample(1:n, nAffected)
		newState[affectedIdxs] .= 0
		return newState
	end
	
	"""Softmax of a vector"""
	softmax(x, eps=0.00001) = exp.(x.+eps) ./ sum(exp.(x.+eps))

	"""
	Normalize the label to put it between 0 and 1
	"""
	function labelNorm(x, ntot)
		return (x + 1) / ntot
	end
	
	function iLabelNorm(x, ntot)
		return (x * ntot) - 1
	end
	
	function norm(x)
		sqrt(sum(x.^2))
	end
	
	"Normalize matrix X along each column"
	function normed(X)
		ns = sqrt.(sum(X .^ 2, dims=1))
		return X ./ ns
	end
		
	"""Given an incomplete seed memory ξ, reach inside X to retrieve most similar memory/combination of memories
	
	If `weighted` is provided, weight the output of the lookup by the vector
	"""
	function updateMem(ξ, X, β)
		lookup = β * transpose(normed(X))*normed(ξ)
		weights = softmax(lookup)
		if any(isnan.(weights))
			weights = softmax(lookup .- max(lookup...))
		end
		return X * weights # Subtract maximum for numerical stability
	end
	
	
	md"""> Memory helper functions"""
end

# ╔═╡ efc1289e-665c-11eb-0b74-a7f88b930522
begin
	abstract type AbstractMemory{T} end
	
	struct ImageMemory{T} <: AbstractMemory{T}
		val::Array{T, 1}
		dims::Array{Int64, 1}
	end
	
	function ImageMemory(img::Array{T, 2}) where T
		ImageMemory{T}(flattenImg(img), [size(img)...])
	end
	
	view(m::ImageMemory) = colorview(Gray, reshape(m.val, m.dims...))
	
	obscure(m::ImageMemory, fracAffected::Real) = ImageMemory(obscureMem(m.val, fracAffected), m.dims)
	
	
	struct LabeledImageMemory{T} <: AbstractMemory{T}
		val::Array{T, 1}
		dims::Array{Int64, 1}
		label::Real
		nLabels::Real
	end

	function createBlankLabeledImageMemory(label::Real, memoryLike::Array{T,1}, dims::Array{Int64,1}; nLabels=10, weight=1) where T
		labels = zeros(nLabels)
		labels[Int(label)] = weight
		LabeledImageMemory{T}(vcat(T.(labels), zero(memoryLike)[nLabels+1:end]), dims, label, nLabels)
	end
	
	function LabeledImageMemory(img::Array{T, 2}, label::Real, nLabels=10) where T
		labelModality = T.(zeros(nLabels))
		labelModality[Int(label)] = 1
		flatImg = flattenImg(img)
		dims = [size(img)...]
		LabeledImageMemory{T}(vcat(labelModality, flatImg), dims, label, nLabels)
	end
	
	function LabeledImageMemory(memVec::Array{T, 1}, dims::Array{Int64, 1}) where T
		LabeledImageMemory{T}(memVec, dims, argmax(memVec[1:10]), 10)
	end
	
	function view(m::LabeledImageMemory{T}) where T
		[getClassname(m.label), colorview(Gray, reshape(m.val[(m.nLabels+1):end], m.dims...))]
	end
	
	function obscure(m::LabeledImageMemory{T}, fracAffected::Real) where T
		LabeledImageMemory(vcat(T.(zeros(m.nLabels)), obscureMem(m.val[(m.nLabels+1):end], fracAffected)), m.dims, m.label, m.nLabels)
	end
	
	md"> Code for the ImageMemory and LabeledImageMemory"
end

# ╔═╡ 0f8357fa-6668-11eb-003b-11b92f1e0f70
begin
	struct MemoryCollection{T}
		val::Array{T, 2}
	end
	
	function MemoryCollection(mems::Array{U, 1}) where U <: AbstractMemory{T} where T
		return MemoryCollection{T}(hcat([m.val for m in mems]...))
	end
	
	function update(ξ::ImageMemory, X::MemoryCollection, β::Real)::ImageMemory
		return ImageMemory(updateMem(ξ.val, X.val, β), ξ.dims)
	end
	
	function update(ξ::LabeledImageMemory, X::MemoryCollection, β::Real)::LabeledImageMemory
		output = updateMem(ξ.val, X.val, β)
		return LabeledImageMemory(output, ξ.dims)
	end
	
	rawMems = withLabel ? [LabeledImageMemory(train_x[:,:,i], train_y[i]) for i in 1:nMemories] : [ImageMemory(train_x[:,:,i]) for i in 1:nMemories]
	X = MemoryCollection(rawMems)
	md"> Create the memories"
end

# ╔═╡ 5fdf7c86-0a7b-11eb-3cd2-fbc1983423d2
begin
	start = showUnseen ? nMemories + 1 : 1
	stop = showUnseen ? n_examples : nMemories
	SeedMemoryIndex = @bind seedIdx Slider(start:stop, show_value=true, default=start)
	md"""**SeedMemoryIndex** = $(SeedMemoryIndex)"""
end

# ╔═╡ ffa9b626-0a6e-11eb-0d32-538c15bc83a4
begin
	sig = (withLabel 
		? LabeledImageMemory(train_x[:,:,seedIdx], train_y[seedIdx])
		: ImageMemory(train_x[:,:,seedIdx]))
	ξ0 = obscure(sig, pctAffected)
	ξ1 = update(ξ0, X, β)
	md"> Calculate the update"
end

# ╔═╡ 0558331a-6760-11eb-0871-f15e627930b5
[view(ξ0), view(ξ1)]

# ╔═╡ d26fbe34-69ba-11eb-368e-45d4af288bb6
begin
	ξ1_clf= ξ1.val[1:10]
	Gray.(ξ1_clf ./ max(ξ1_clf...))
end

# ╔═╡ 93160af6-6972-11eb-0640-4309a0702351
md"""
But obviously, classifying complex images like CIFAR is not something you want to do with a pure associative memory.
"""

# ╔═╡ def095f4-67bf-11eb-1bcc-a55a68c884d2
md"""
### Generating an Image from a Label

Now we have a label inside of our memory -- can we ask the model to generate images for us?
"""

# ╔═╡ c375c2a0-6976-11eb-12b8-d55f2f456cda
begin
	Beta2 = @bind βgen Slider(0.001:1:400, show_value=true)
	md"""**Beta** = $(Beta2)"""
end

# ╔═╡ 899b669c-69bd-11eb-100c-db8191c320a2
md"""
Select a class to generate from the list below:
"""

# ╔═╡ 576e36be-69bc-11eb-1a0f-b14e301897ad
begin
	classOptions = [getClassname(i) for i in 1:10]
	htmlClassOptions = ["<option value=\"$(classOptions[c])\">$(classOptions[c])</option>" for c = 1:length(classOptions)]
	
	optionContent = join(htmlClassOptions, "")
	@bind labelName HTML("""
	<select>
		$(optionContent)
	</select>
	""")
end

# ╔═╡ c82cf396-6975-11eb-10e3-d94ab6db4050
begin
	label = getLabel(labelName)
	ξblank = createBlankLabeledImageMemory(label, ξ0.val, ξ0.dims)
	ξgen = update(ξblank, X, βgen)
	view(ξgen)
end 

# ╔═╡ Cell order:
# ╟─b105d9c2-0a79-11eb-2f6f-4b12f9498f89
# ╟─08ec4722-6761-11eb-1b5c-d598cc7abaea
# ╟─70d5be76-0a5d-11eb-2626-6799926deaeb
# ╠═df4e3e08-0a5d-11eb-331c-59c2cd60bb4a
# ╟─f08135ee-0a77-11eb-2caa-ab85b93ddb38
# ╠═efc1289e-665c-11eb-0b74-a7f88b930522
# ╠═0f8357fa-6668-11eb-003b-11b92f1e0f70
# ╟─e37dcd7e-0a79-11eb-1273-2f35d70a42a1
# ╟─5fdf7c86-0a7b-11eb-3cd2-fbc1983423d2
# ╟─ed96a526-0a68-11eb-0dd8-33787eab6779
# ╟─b2ff9d5a-0a6d-11eb-0803-792ae4dd0997
# ╠═7de67790-0a6f-11eb-2b9f-91ec6a0035e9
# ╟─82f2fbe0-0a7a-11eb-2faf-5730af229577
# ╟─ffa9b626-0a6e-11eb-0d32-538c15bc83a4
# ╟─0558331a-6760-11eb-0871-f15e627930b5
# ╟─bad039e0-69ba-11eb-0f3b-4d4e747f9142
# ╟─d26fbe34-69ba-11eb-368e-45d4af288bb6
# ╟─9d4ffdc6-0a7a-11eb-3e52-4f7c39e6f8d2
# ╟─dea5fafa-0a7a-11eb-2db7-a16b04aa4b65
# ╟─a74409a2-0a7b-11eb-2b15-b51ae49b13fb
# ╟─b06e8c6c-6653-11eb-1e2c-6948e68e20fb
# ╟─e235e2b6-668c-11eb-2cec-879584be1e27
# ╟─abf2e53a-6652-11eb-1cbf-8d0cef72a087
# ╟─715eb624-6652-11eb-337a-4b6fe9950763
# ╟─93160af6-6972-11eb-0640-4309a0702351
# ╟─def095f4-67bf-11eb-1bcc-a55a68c884d2
# ╟─c375c2a0-6976-11eb-12b8-d55f2f456cda
# ╟─899b669c-69bd-11eb-100c-db8191c320a2
# ╟─576e36be-69bc-11eb-1a0f-b14e301897ad
# ╟─c82cf396-6975-11eb-10e3-d94ab6db4050
