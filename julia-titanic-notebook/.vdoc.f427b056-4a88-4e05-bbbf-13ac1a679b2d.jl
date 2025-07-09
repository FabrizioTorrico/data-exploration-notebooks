#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
using Pkg
Pkg.activate(".")
Pkg.add(["IJulia", "DataFrames", "CSV", "CairoMakie", "StatsBase",
         "Statistics", "MLJ", "MLJModels", "HypothesisTests",
         "Distributions", "Missings", "CategoricalArrays", "AlgebraOfGraphics", "Chain"])
#
#
#
#| _cell_guid: 80643cb5-64f3-4180-92a9-2f8e83263ac6
#| _kg_hide-input: true
#| _uuid: 33d54abf387474bce3017f1fc3832493355010c0
#| tags: []
import DataFrames as DF
import CSV
import CairoMakie as Makie
import AlgebraOfGraphics as AoG
import Statistics as Stats
import StatsBase
import Chain: @chain
import Random: shuffle
import IJulia


#
#
#
readdir("./input/")
#
#
#
#
#
#
#
#
#
#
#
## Importing the datasets
using CSV

train = CSV.read("./input/train.csv", DF.DataFrame)
test = CSV.read("./input/test.csv", DF.DataFrame);
#
#
#
#
#
#
#
DF.first(train, 5)
print(train.Pclass)
#
#
#
@chain train begin
    DF.dropmissing(:Age) # Drop rows with missing Age
    DF.groupby(:Sex)
    DF.combine(:Age => minimum => :MinAge)
end
#
#
#
DF.describe(train, :eltype)
#
#
#
#
#
#
#
#
#
#
#
DF.first(train[shuffle(1:DF.nrow(train))[1:5], :], 5)
#
#
#
#
#
DF.first(test[shuffle(1:DF.nrow(test))[1:5], :], 5)
#
#
#
#
#
println("The shape of the train data is (row, column): $(size(train))")
println("Train dataset info:")
DF.describe(train)


println("The shape of the test data is (row, column): $(size(test))")
println("Test dataset info:")
DF.describe(test)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| _cell_guid: bf19c831-fbe0-49b6-8bf8-d7db118f40b1
#| _kg_hide-input: true
#| _uuid: 5a0593fb4564f0284ca7fdf5c006020cb288db95
#| execution: {iopub.execute_input: '2021-06-26T16:35:08.956119Z', iopub.status.busy: '2021-06-26T16:35:08.955538Z', iopub.status.idle: '2021-06-26T16:35:08.973222Z', shell.execute_reply: '2021-06-26T16:35:08.972151Z', shell.execute_reply.started: '2021-06-26T16:35:08.956072Z'}
DF.describe(train, :nmissing, :eltype)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| _kg_hide-input: true
#| execution: {iopub.execute_input: '2021-06-26T16:35:08.975451Z', iopub.status.busy: '2021-06-26T16:35:08.974927Z', iopub.status.idle: '2021-06-26T16:35:08.98326Z', shell.execute_reply: '2021-06-26T16:35:08.982644Z', shell.execute_reply.started: '2021-06-26T16:35:08.975205Z'}
function missing_percentage(df::DF.DataFrame)
    """This function takes a DataFrame as input and returns total missing values and percentages"""
    missing_counts = [count(ismissing, df[!, col]) for col in DF.names(df)]
    missing_pct = round.(missing_counts ./ DF.nrow(df) .* 100, digits=2)

    # Create result DataFrame
    result = DF.DataFrame(
        Column = DF.names(df),
        Total = missing_counts,
        Percent = missing_pct
    )

    # Sort by total missing values (descending)
    return DF.sort(result, :Total, rev=true)
end
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:35:09.092256Z', iopub.status.busy: '2021-06-26T16:35:09.09199Z', iopub.status.idle: '2021-06-26T16:35:09.108063Z', shell.execute_reply: '2021-06-26T16:35:09.107054Z', shell.execute_reply.started: '2021-06-26T16:35:09.092212Z'}
missing_percentage(train)
#
#
#
#
#
#| _cell_guid: 073ef91b-e401-47a1-9b0a-d08ad710abce
#| _kg_hide-input: true
#| _uuid: 1ec1de271f57c9435ce111261ba08c5d6e34dbcb
#| execution: {iopub.execute_input: '2021-06-26T16:35:09.208229Z', iopub.status.busy: '2021-06-26T16:35:09.207968Z', iopub.status.idle: '2021-06-26T16:35:09.221423Z', shell.execute_reply: '2021-06-26T16:35:09.220732Z', shell.execute_reply.started: '2021-06-26T16:35:09.208186Z'}
missing_percentage(test)
#
#
#
#
#
#
#
#
#
#| _kg_hide-input: true
#| execution: {iopub.execute_input: '2021-06-26T16:35:09.223175Z', iopub.status.busy: '2021-06-26T16:35:09.222681Z', iopub.status.idle: '2021-06-26T16:35:09.230671Z', shell.execute_reply: '2021-06-26T16:35:09.229793Z', shell.execute_reply.started: '2021-06-26T16:35:09.223128Z'}
function percent_value_counts(df::DF.DataFrame, feature::Symbol)
    """This function takes a dataframe and a column and finds the percentage of the value_counts"""

    # Count values including missing
    counts = DF.combine(DF.groupby(df, feature), DF.nrow => :Total)

    # Calculate percentages
    counts.Percent = round.(counts.Total ./ DF.nrow(df) .* 100, digits=2)

    # Sort by total count (descending)
    return DF.sort(counts, :Total, rev=true)
end
#
#
#
#| _kg_hide-input: true
#| execution: {iopub.execute_input: '2021-06-26T16:35:09.236974Z', iopub.status.busy: '2021-06-26T16:35:09.236548Z', iopub.status.idle: '2021-06-26T16:35:09.254321Z', shell.execute_reply: '2021-06-26T16:35:09.253654Z', shell.execute_reply.started: '2021-06-26T16:35:09.236929Z'}
percent_value_counts(train, :Embarked)
#
#
#
#
#
#
#
#| _cell_guid: 000ebdd7-ff57-48d9-91bf-a29ba79f1a1c
#| _kg_hide-input: true
#| _uuid: 6b9cb050e9dae424bb738ba9cdf3c84715887fa3
#| execution: {iopub.execute_input: '2021-06-26T16:35:09.276102Z', iopub.status.busy: '2021-06-26T16:35:09.275649Z', iopub.status.idle: '2021-06-26T16:35:09.292037Z', shell.execute_reply: '2021-06-26T16:35:09.291163Z', shell.execute_reply.started: '2021-06-26T16:35:09.275879Z'}
train[ismissing.(train.Embarked), :]
#
#
#
#
#
#| _cell_guid: bf257322-0c9c-4fc5-8790-87d8c94ad28a
#| _kg_hide-input: true
#| _uuid: ad15052fe6cebe37161c6e01e33a5c083dc2b558
#| execution: {iopub.execute_input: '2021-06-26T16:35:09.293919Z', iopub.status.busy: '2021-06-26T16:35:09.293564Z', iopub.status.idle: '2021-06-26T16:35:09.866643Z', shell.execute_reply: '2021-06-26T16:35:09.865701Z', shell.execute_reply.started: '2021-06-26T16:35:09.293817Z'}
fig = Makie.Figure()

# Prepare data for plotting
train_clean = DF.dropmissing(train, [:Embarked, :Fare, :Pclass])
test_clean = DF.dropmissing(test, [:Embarked, :Fare, :Pclass])

# Create mapping for embarked ports to numbers
unique_categories = unique(train_clean.Embarked)
category_to_index = Dict(category => i for (i, category) in enumerate(unique_categories))
# Convert categorical to numeric
train_clean.Embarked_num = [category_to_index[port] for port in train_clean.Embarked]
test_clean.Embarked_num = [category_to_index[port] for port in test_clean.Embarked]

# Training set boxplot
ax1 = Makie.Axis(fig[1, 1],
    title = "Training Set",
    xlabel = "Embarked",
    ylabel = "Fare",
    xticks = (1:3, unique_categories)
)

ax2 = Makie.Axis(fig[1, 2],
    title = "Test Set",
    xlabel = "Embarked",
    ylabel = "Fare",
    xticks = (1:3, unique_categories)
)

Makie.boxplot!(ax2, test_clean.Embarked_num, test_clean.Fare,
           dodge = test_clean.Pclass,
           color = test_clean.Pclass)
Makie.boxplot!(ax1, train_clean.Embarked_num, train_clean.Fare,
           dodge = train_clean.Pclass,
           color = train_clean.Pclass)

fig
#
#
#
#
#
#| _cell_guid: 2f5f3c63-d22c-483c-a688-a5ec2a477330
#| _kg_hide-input: true
#| _uuid: 52e51ada5dfeb700bf775c66e9307d6d1e2233de
#| execution: {iopub.execute_input: '2021-06-26T16:35:09.868523Z', iopub.status.busy: '2021-06-26T16:35:09.868016Z', iopub.status.idle: '2021-06-26T16:35:09.874135Z', shell.execute_reply: '2021-06-26T16:35:09.873022Z', shell.execute_reply.started: '2021-06-26T16:35:09.868249Z'}
#| scrolled: true
## Replacing the null values in the Embarked column with the mode.
train.Embarked = coalesce.(train.Embarked, "C");
#
#
#
#
#
#
#
#| _cell_guid: e76cd770-b498-4444-b47a-4ac6ae63193b
#| _kg_hide-input: true
#| _uuid: b809a788784e2fb443457d7ef4ca17a896bf58b4
#| execution: {iopub.execute_input: '2021-06-26T16:35:09.876171Z', iopub.status.busy: '2021-06-26T16:35:09.875621Z', iopub.status.idle: '2021-06-26T16:35:09.886193Z', shell.execute_reply: '2021-06-26T16:35:09.885088Z', shell.execute_reply.started: '2021-06-26T16:35:09.875859Z'}
#| scrolled: true
println("Train Cabin missing: $(count(ismissing, train.Cabin) / DF.nrow(train))")
println("Test Cabin missing: $(count(ismissing, test.Cabin) / DF.nrow(test))")
#
#
#
#
#
#
#
#
#
#
#
#| _kg_hide-input: true
#| _uuid: 8ff7b4f88285bc65d72063d7fdf8a09a5acb62d3
#| execution: {iopub.execute_input: '2021-06-26T16:35:09.888377Z', iopub.status.busy: '2021-06-26T16:35:09.88784Z', iopub.status.idle: '2021-06-26T16:35:09.902296Z', shell.execute_reply: '2021-06-26T16:35:09.901697Z', shell.execute_reply.started: '2021-06-26T16:35:09.888114Z'}
survivors = train.Survived
DF.select!(train, DF.Not(:Survived))  # Remove Survived column
all_data = vcat(train, test)

all_data.Cabin = coalesce.(all_data.Cabin, "N");
#
#
#
#
#
#| _cell_guid: 87995359-8a77-4e38-b8bb-e9b4bdeb17ed
#| _kg_hide-input: true
#| _uuid: c1e9e06eb7f2a6eeb1a6d69f000217e7de7d5f25
#| execution: {iopub.execute_input: '2021-06-26T16:35:09.904181Z', iopub.status.busy: '2021-06-26T16:35:09.903766Z', iopub.status.idle: '2021-06-26T16:35:09.909654Z', shell.execute_reply: '2021-06-26T16:35:09.908573Z', shell.execute_reply.started: '2021-06-26T16:35:09.904014Z'}
all_data.Cabin = [string(cabin[1]) for cabin in all_data.Cabin];
#
#
#
#
#
#| _kg_hide-input: true
#| execution: {iopub.execute_input: '2021-06-26T16:35:09.91156Z', iopub.status.busy: '2021-06-26T16:35:09.911098Z', iopub.status.idle: '2021-06-26T16:35:09.928945Z', shell.execute_reply: '2021-06-26T16:35:09.928025Z', shell.execute_reply.started: '2021-06-26T16:35:09.911398Z'}
percent_value_counts(all_data, :Cabin)
#
#
#
#
#
#
#
#| _kg_hide-input: true
#| execution: {iopub.execute_input: '2021-06-26T16:35:09.930774Z', iopub.status.busy: '2021-06-26T16:35:09.930283Z', iopub.status.idle: '2021-06-26T16:35:09.942122Z', shell.execute_reply: '2021-06-26T16:35:09.941067Z', shell.execute_reply.started: '2021-06-26T16:35:09.930532Z'}
@chain all_data begin
    DF.dropmissing(:Fare)
    DF.groupby(:Cabin)
    DF.combine(:Fare => Stats.mean => :Mean_Fare)
    DF.sort(:Mean_Fare)
end
#
#
#
#
#
#| _kg_hide-input: true
#| _uuid: a466da29f1989fa983147faf9e63d18783468567
#| execution: {iopub.execute_input: '2021-06-26T16:35:09.943855Z', iopub.status.busy: '2021-06-26T16:35:09.943364Z', iopub.status.idle: '2021-06-26T16:35:09.952677Z', shell.execute_reply: '2021-06-26T16:35:09.952057Z', shell.execute_reply.started: '2021-06-26T16:35:09.943627Z'}
function cabin_estimator(fare::Union{Float64, Missing})
    """Grouping cabin feature by the first letter based on fare"""
    # Handle missing values
    if ismissing(fare)
        return "N"  # Default cabin for missing fare
    end
    
    if fare < 16
        return "G"
    elseif 16 ≤ fare < 27
        return "F"
    elseif 27 ≤ fare < 38
        return "T"
    elseif 38 ≤ fare < 47
        return "A"
    elseif 47 ≤ fare < 53
        return "E"
    elseif 53 ≤ fare < 54
        return "D"
    elseif 54 ≤ fare < 116
        return "C"
    else
        return "B"
    end
end
#
#
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:35:09.95455Z', iopub.status.busy: '2021-06-26T16:35:09.954083Z', iopub.status.idle: '2021-06-26T16:35:09.96302Z', shell.execute_reply: '2021-06-26T16:35:09.962357Z', shell.execute_reply.started: '2021-06-26T16:35:09.95437Z'}
with_N = all_data[all_data.Cabin .== "N", :]
without_N = all_data[all_data.Cabin .!= "N", :];
#
#
#
#| _kg_hide-input: true
#| _uuid: 1c646b64c6e062656e5f727d5499266f847c4832
#| execution: {iopub.execute_input: '2021-06-26T16:35:09.965179Z', iopub.status.busy: '2021-06-26T16:35:09.96464Z', iopub.status.idle: '2021-06-26T16:35:09.981536Z', shell.execute_reply: '2021-06-26T16:35:09.980705Z', shell.execute_reply.started: '2021-06-26T16:35:09.964885Z'}
with_N.Cabin = cabin_estimator.(with_N.Fare)

# Combine back together
all_data = vcat(with_N, without_N)

# Sort by PassengerId
DF.sort!(all_data, :PassengerId)

# Separate train and test
train = all_data[1:891, :]
test = all_data[892:end, :]

# Add back survival information
train.Survived = survivors;
#
#
#
#
#
#
#
#
#
test[ismissing.(test.Fare), :]
#
#
#
#
#
#| _cell_guid: e742aa76-b6f8-4882-8bd6-aa10b96f06aa
#| _kg_hide-input: true
#| _uuid: f1dc8c6c33ba7df075ee608467be2a83dc1764fd
#| execution: {iopub.execute_input: '2021-06-26T16:35:10.002749Z', iopub.status.busy: '2021-06-26T16:35:10.002232Z', iopub.status.idle: '2021-06-26T16:35:10.012662Z', shell.execute_reply: '2021-06-26T16:35:10.011431Z', shell.execute_reply.started: '2021-06-26T16:35:10.00248Z'}
missing_value = @chain test begin
    DF.subset(:Pclass => x -> x .== 3, :Embarked => x -> x .== "S", :Sex => x -> x .== "male")
    _.Fare
    skipmissing
    Stats.mean
end

test.Fare = coalesce.(test.Fare, missing_value);
#
#
#
#
#
#
#
#
#
#| _cell_guid: 8ff25fb3-7a4a-4e06-b48f-a06b8d844917
#| _kg_hide-input: true
#| _uuid: c356e8e85f53a27e44b5f28936773a289592c5eb
#| execution: {iopub.execute_input: '2021-06-26T16:35:10.014347Z', iopub.status.busy: '2021-06-26T16:35:10.014023Z', iopub.status.idle: '2021-06-26T16:35:10.024214Z', shell.execute_reply: '2021-06-26T16:35:10.023404Z', shell.execute_reply.started: '2021-06-26T16:35:10.014284Z'}
println("Train age missing value: $(round(count(ismissing, train.Age) / DF.nrow(train) * 100, digits=2))%")
println("Test age missing value: $(round(count(ismissing, test.Age) / DF.nrow(test) * 100, digits=2))%")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
Makie.set_theme!(Makie.theme_light())
#
#
#
fig = Makie.Figure()
ax = Makie.Axis(fig[1, 1], 
    title = "Survived/Non-Survived Passenger Gender Distribution",
    xlabel = "Sex",
    ylabel = "% of passenger survived",
    xticks= (1:2, ["Male", "Female"]),
    
)

# Calculate survival rates by gender
survival_by_sex = @chain train begin
    DF.groupby(:Sex)
    DF.combine(:Survived => Stats.mean => :survival_rate)
    DF.sort(:Sex, rev=true)  # Female first
end

# Create elegant barplot
Makie.barplot!(ax, 1:2, survival_by_sex.survival_rate, 
           color = ["green", "pink"])

fig

#
#
#
#
#
#
fig = Makie.Figure()
ax = Makie.Axis(fig[1, 1],
    title = "Passenger Gender Distribution - Survived vs Not-survived",
    xlabel = "Sex",
    ylabel = "# of Passenger Survived",
    xticks = (1:2, ["Male", "Female"])
)

# Count data for grouped bar chart
count_data = @chain train begin
    DF.groupby([:Sex, :Survived])
    DF.combine(DF.nrow => :count)
    DF.unstack(:Survived, :count, fill=0)
end

# Create grouped bar chart
counts = [count_data[1, 2], count_data[1, 3], count_data[2, 2], count_data[2, 3]]


Makie.barplot!(ax, [1, 1, 2, 2], counts,
           dodge = [1, 2, 1,2],
           color = ["gray", "green", "gray", "green"])



# Add legend
Makie.Legend(fig[1, 2], 
    [Makie.PolyElement(color = "gray"), Makie.PolyElement(color = "green")],
    ["Not Survived", "Survived"],
    "Survival Status")

fig
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
fig = Makie.Figure()
ax = Makie.Axis(fig[1, 1],
    title = "Passenger Class Distribution - Survival Percentage",
    xlabel = "Passenger Class",
    ylabel = "Percentage",
    titlesize = 20,
    xlabelsize = 16,
    ylabelsize = 16,
    xticks=(1:3, ["1st Class", "2nd Class", "3rd Class"])
)

# Calculate percentages by class
class_survival = @chain train begin
    DF.groupby([:Pclass, :Survived])
    DF.combine(DF.nrow => :count)
    DF.unstack(:Survived, :count, fill=0)
end

no_survived = class_survival[:, 2]
yes_survived = class_survival[:, 3]
total_by_class = no_survived + yes_survived

survived_percentage = (yes_survived ./ total_by_class) * 100
not_survived_percentage = (no_survived ./ total_by_class) * 100

flatten = vcat(not_survived_percentage ,survived_percentage)

Makie.barplot!(ax, [1, 2, 3, 1, 2, 3], flatten, stack=[1, 2, 3, 1, 2, 3], color = ["red", "red", "red", "green", "green", "green"], strokewidth = 1, strokecolor = :black)

# Add legend
Makie.Legend(fig[1, 2],
    [Makie.PolyElement(color = "#F44336"), Makie.PolyElement(color = "#4CAF50")],
    ["Not Survived", "Survived"],
    "Survival Status")

fig
#
#
#
Makie.barplot([1, 2, 3], survived_percentage, axis=(xticks=(1:3, ["1st Class", "2nd Class", "3rd Class"]), title = "Passenger Class Distribution - Survived vs Non-Survived"), color=["brown", "orange", "green"])
#
#
#
#
#
#
#
#

fig = Makie.Figure(
    title = "Passenger Class Distribution - Survived vs Non-Survived",
    xlabel = "Passenger Class",
    ylabel = "Density of Passenger Survived",
    
) # Adjust figure size as needed
ax =  Makie.Axis(fig[1, 1], xticks = ([1, 2, 3], ["Upper", "Middle", "Lower"]))           

d1 = Makie.density!(ax, train.Pclass[train.Survived .== 0], color = (:gray, 0.2), strokecolor=:gray, strokewidth=2)
d2= Makie.density!(ax, train.Pclass[train.Survived .== 1], color = (:green, 0.2), strokecolor=:green, strokewidth=2)

Makie.axislegend(ax,
    [d1, d2],
    ["Not Survived", "Survived"],
    "Survival Status")

fig
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
fig = Makie.Figure()

ax = Makie.Axis(fig[1, 1],
    title = "Fare Distribution - Survived vs Non-Survived",
    xlabel = "Fare",
    ylabel = "Density of Passenger Survived",
)
 
d1 = Makie.density!(ax, train.Fare[train.Survived .== 0], color = (:gray, 0.2), strokecolor=:gray, strokewidth=2)
d2 = Makie.density!(ax, train.Fare[train.Survived .== 1], color = (:green, 0.2), strokecolor=:green, strokewidth=2)

Makie.axislegend(ax,
    [d1, d2],
    ["Not Survived", "Survived"],
    "Survival Status")
fig
#
#
#
#
#
#
#
#
train[train.Fare .> 280, :]
#
#
#
#
#
#
#
#
#
#
#

fig = Makie.Figure()

ax = Makie.Axis(fig[1, 1], title = "Age Distribution - Survived vs Non-Survived",
    xlabel = "Age",
    ylabel = "Density of Passenger Survived")


# clean missing first
clean_train =  DF.dropmissing(train, :Age)
not_survived = clean_train.Age[clean_train.Survived .== 0]
survived = clean_train.Age[clean_train.Survived .== 1]

d1 = Makie.density!(ax, not_survived, color = (:gray, 0.2), strokecolor=:gray, strokewidth=2)
d2 = Makie.density!(ax, survived, color = (:green, 0.2), strokecolor=:green, strokewidth=2)

Makie.axislegend(ax,
    [d1, d2],
    ["Not Survived", "Survived"],
    "Survival Status")

fig
#
#
#
#
#
#
#
#
#
#
#
#
#

fig = Makie.Figure(title="Survived by Sex and Age")

# Create subplots for each combination

for (i, (sex, survived)) in enumerate(Iterators.product(["female", "male"], [0, 1]))

    ax = Makie.Axis(fig[div(i - 1, 2) + 1, i % 2 + 1],
        title = "$sex $(survived == 1 ? "Survived" : "Not Survived")",
        xlabel = "Age",
        ylabel = "Count"
    )
    
    subset_data = train[(train.Sex .== sex) .& (train.Survived .== survived) .& .!ismissing.(train.Age), :]
    
    if DF.nrow(subset_data) > 0
        Makie.hist!(ax, subset_data.Age, bins = 20, 
                color = survived == 1 ? "green" : "gray",
                strokewidth = 1, strokecolor = :white)
    end
end

fig
#
#
#
#
#
fig = Makie.Figure(title="Survived by Sex and Age")

# Create subplots for each combination
for (i, (sex, embarked)) in enumerate(Iterators.product(["female", "male"], ["S", "C", "Q"]))

    ax = Makie.Axis(fig[div(i - 1, 2) + 1, i % 2 + 1],
        title = "$sex $embarked",
    )

    subset_data = train[(train.Sex .== sex) .& (train.Embarked .== embarked) .& .!ismissing.(train.Age), :]

    for (survived) in [0, 1]
        subset_survived = subset_data[(subset_data.Survived .== survived), :]
        println("Length of subset: $(DF.nrow(subset_survived))")

        if DF.nrow(subset_data) > 0
             Makie.hist!(ax, subset_survived.Age, 
                        bins = 20,
                        color = survived == 1 ? (:green, 0.5) : (:gray, 0.5),
                        strokewidth = 1, 
                        strokecolor = :white,
                        label = survived == 1 ? "Survived" : "Not Survived"
                    )
        end
    end
end


Makie.Legend(fig[1, 3], 
    [Makie.PolyElement(color = (:gray, 0.7)), 
     Makie.PolyElement(color = (:green, 0.7))],
    ["Not Survived", "Survived"],
    "Survival Status"
)

fig
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
fig = Makie.Figure(resolution = (1000, 600))

ax_m = Makie.Axis(fig[1, 1],
    title = "Male", 
    xlabel = "Fare",
    ylabel = "Age")
# Female subplot
ax_f = Makie.Axis(fig[1, 2], 
    title = "Female",
    xlabel = "Fare",
    ylabel = "Age")

female_data = train[(train.Sex .== "female") .& .!ismissing.(train.Age), :]
male_data = train[(train.Sex .== "male") .& .!ismissing.(train.Age), :]


Makie.scatter!(ax_m, male_data.Fare, male_data.Age,
           color = [s == 1 ? "green" : "gray" for s in male_data.Survived],
           strokewidth=1, strokecolor="white", markersize=14)
Makie.scatter!(ax_f, female_data.Fare, female_data.Age,
           color = [s == 1 ? "green" : "gray" for s in female_data.Survived],
           strokewidth=1, strokecolor="white", markersize=14)


# Add legend
Makie.Legend(fig[1, 3],
    [Makie.MarkerElement(color = "gray", marker = :circle), 
     Makie.MarkerElement(color = "green", marker = :circle)],
    ["Not Survived", "Survived"],
    "Survived")

Makie.Label(fig[0, :], "Survived by Sex, Fare and Age")
fig
#
#
#
#
#
#
#
#
train = train[train.Fare .< 500, :]

fig = Makie.Figure(size = (800, 600))
ax = Makie.Axis(fig[1, 1],
    title = "Parents/Children Survival Rate",
    xlabel = "Number of Parents/Children",
    ylabel = "Survival Rate",
)

parch_survival = @chain train_clean begin
    DF.groupby(:Parch)
    DF.combine(
        :Survived => Stats.mean => :survival_rate,
        :Survived => Stats.std => :std_dev,
        :Survived => length => :count
    )
end

parch_survival.std_error = parch_survival.std_dev ./ sqrt.(parch_survival.count)

Makie.scatterlines!(ax, parch_survival.Parch, parch_survival.survival_rate,
    color = "#2196F3", 
    linewidth = 3,
    markersize = 8
)

error = Makie.errorbars!(ax, parch_survival.Parch, parch_survival.survival_rate, 
    parch_survival.std_error,
    color = "blue",
    linewidth = 2,
    whiskerwidth = 8
)

Makie.Legend(fig[1, 2],
    [Makie.PolyElement(color = "#2196F3"), Makie.PolyElement(color = "blue")],
    ["Survival Rate", "Standard Error"],
    "Legend"
)
fig
#
#
#
#
#
# sns.factorplot(x =  "SibSp", y = "Survived", data = train,kind = "point",size = 8)
# plt.title('Factorplot of Sibilings/Spouses survived', fontsize = 25)
# plt.subplots_adjust(top=0.85)

fig = Makie.Figure(size = (800, 600))
ax = Makie.Axis(fig[1, 1],
    title = "Siblings/Spouses Survival Rate",
    xlabel = "Number of Siblings/Spouses",
    ylabel = "Survival Rate",
)

sibsp_survival = @chain train_clean begin
    DF.groupby(:SibSp)
    DF.combine(
        :Survived => Stats.mean => :survival_rate,
        :Survived => Stats.std => :std_dev,
        :Survived => length => :count
    )
end

sibsp_survival.std_error = sibsp_survival.std_dev ./ sqrt.(sibsp_survival.count)

Makie.scatterlines!(ax, sibsp_survival.SibSp, sibsp_survival.survival_rate,
    color = "#2196F3", 
    linewidth = 3,
    markersize = 8
)

error = Makie.errorbars!(ax, sibsp_survival.SibSp, sibsp_survival.survival_rate, 
    sibsp_survival.std_error,
    color = "blue",
    linewidth = 2,
    whiskerwidth = 8
)

Makie.Legend(fig[1, 2],
    [Makie.PolyElement(color = "#2196F3"), Makie.PolyElement(color = "blue")],
    ["Survival Rate", "Standard Error"],
    "Legend"
)
fig
#
#
#
#
#
train.Sex = [sex == "female" ? 0 : 1 for sex in train.Sex]
test.Sex = [sex == "female" ? 0 : 1 for sex in test.Sex];
#
#
#
#
#
#
#
#
#
#
#
#
#
DF.describe(train)
#
#
#
categorical_cols = [col for col in names(train) if eltype(train[!, col]) <: Union{String, AbstractString}]
DF.describe(train[!, categorical_cols])
#
#
#
survived_summary = @chain train begin
    DF.select(DF.names(train, Number)...)
    DF.groupby(:Survived)
    DF.combine(DF.All() .=> Stats.mean)
end
#
#
#
sex_summary = @chain train begin
    DF.select(DF.names(train, Number)...)
    DF.groupby(:Sex)
    DF.combine(DF.All() .=> Stats.mean)
end
#
#
#
class_summary = @chain train begin
    DF.select(DF.names(train, Number)...)
    DF.groupby(:Pclass)
    DF.combine(DF.All() .=> Stats.mean)
end
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
train_numeric = DF.select(train, DF.names(train, Number)...)
corr_matrix = Stats.cor(Stats.Matrix(train_numeric))

corr_df = DF.DataFrame(corr_matrix, DF.names(train, Number))
#
#
#
#
DF.sort(DF.DataFrame(
    Variable = DF.names(corr_df),
    Correlation = abs.(corr_matrix[:, 1])
), [:Correlation], rev=true)
#
#
#
#
#
#
corr_matrix = Stats.cor(Stats.Matrix(train_numeric), train_numeric.Survived) .^ 2

corr_matrix = DF.sort(DF.DataFrame(
    Variable = DF.names(train_numeric),
    Correlation = abs.(corr_matrix[:, 1])
), [:Correlation], rev=true)
#
#
#
#
#
corr_matrix = Stats.cor(Stats.Matrix(train_numeric))
corr_matrix = DF.DataFrame(
    Correlation )
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| _kg_hide-input: true
#| _uuid: abd034cffc591bf1ef2b4a8ed3e5a65eb133d61e
#| execution: {iopub.execute_input: '2021-06-26T16:35:15.771771Z', iopub.status.busy: '2021-06-26T16:35:15.771345Z', iopub.status.idle: '2021-06-26T16:35:15.783362Z', shell.execute_reply: '2021-06-26T16:35:15.782301Z', shell.execute_reply.started: '2021-06-26T16:35:15.771603Z'}
male_mean = train[train['Sex'] == 1].Survived.mean()

female_mean = train[train['Sex'] == 0].Survived.mean()
print ("Male survival mean: " + str(male_mean))
print ("female survival mean: " + str(female_mean))

print ("The mean difference between male and female survival rate: " + str(female_mean - male_mean))
#
#
#
#
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:35:15.785359Z', iopub.status.busy: '2021-06-26T16:35:15.784861Z', iopub.status.idle: '2021-06-26T16:35:15.815921Z', shell.execute_reply: '2021-06-26T16:35:15.815302Z', shell.execute_reply.started: '2021-06-26T16:35:15.785103Z'}
# separating male and female dataframe.
import random
male = train[train['Sex'] == 1]
female = train[train['Sex'] == 0]

## empty list for storing mean sample
m_mean_samples = []
f_mean_samples = []

for i in range(50):
    m_mean_samples.append(np.mean(random.sample(list(male['Survived']),50,)))
    f_mean_samples.append(np.mean(random.sample(list(female['Survived']),50,)))


# Print them out
print (f"Male mean sample mean: {round(np.mean(m_mean_samples),2)}")
print (f"Male mean sample mean: {round(np.mean(f_mean_samples),2)}")
print (f"Difference between male and female mean sample mean: {round(np.mean(f_mean_samples) - np.mean(m_mean_samples),2)}")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| _cell_guid: d30d71c1-55bc-41c8-8536-9909d9f02538
#| _kg_hide-input: true
#| _uuid: cb17c6f59bb2123cbf2cbc9c282b4d70ee283a86
#| execution: {iopub.execute_input: '2021-06-26T16:35:15.817993Z', iopub.status.busy: '2021-06-26T16:35:15.817477Z', iopub.status.idle: '2021-06-26T16:35:15.832377Z', shell.execute_reply: '2021-06-26T16:35:15.831471Z', shell.execute_reply.started: '2021-06-26T16:35:15.817745Z'}
# Creating a new colomn with a
train['name_length'] = [len(i) for i in train.Name]
test['name_length'] = [len(i) for i in test.Name]

def name_length_group(size):
    a = ''
    if (size <=20):
        a = 'short'
    elif (size <=35):
        a = 'medium'
    elif (size <=45):
        a = 'good'
    else:
        a = 'long'
    return a


train['nLength_group'] = train['name_length'].map(name_length_group)
test['nLength_group'] = test['name_length'].map(name_length_group)

## Here "map" is python's built-in function.
## "map" function basically takes a function and
## returns an iterable list/tuple or in this case series.
## However,"map" can also be used like map(function) e.g. map(name_length_group)
## or map(function, iterable{list, tuple}) e.g. map(name_length_group, train[feature]]).
## However, here we don't need to use parameter("size") for name_length_group because when we
## used the map function like ".map" with a series before dot, we are basically hinting that series
## and the iterable. This is similar to .append approach in python. list.append(a) meaning applying append on list.


## cuts the column by given bins based on the range of name_length
#group_names = ['short', 'medium', 'good', 'long']
#train['name_len_group'] = pd.cut(train['name_length'], bins = 4, labels=group_names)
#
#
#
#
#
#
#
#| _cell_guid: ded64d5f-43de-4a9e-b9c5-ec4d2869387a
#| _kg_hide-input: true
#| _uuid: 9c23229f7d06a1303a04b4a81c927453686ffec9
#| execution: {iopub.execute_input: '2021-06-26T16:35:15.833953Z', iopub.status.busy: '2021-06-26T16:35:15.833501Z', iopub.status.idle: '2021-06-26T16:35:15.842414Z', shell.execute_reply: '2021-06-26T16:35:15.841468Z', shell.execute_reply.started: '2021-06-26T16:35:15.83376Z'}
## get the title from the name
train["title"] = [i.split('.')[0] for i in train.Name]
train["title"] = [i.split(',')[1] for i in train.title]
## Whenever we split like that, there is a good change that we will end up with while space around our string values. Let's check that.
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:35:15.84422Z', iopub.status.busy: '2021-06-26T16:35:15.843818Z', iopub.status.idle: '2021-06-26T16:35:15.853522Z', shell.execute_reply: '2021-06-26T16:35:15.852642Z', shell.execute_reply.started: '2021-06-26T16:35:15.84407Z'}
print(train.title.unique())
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:35:15.855322Z', iopub.status.busy: '2021-06-26T16:35:15.854858Z', iopub.status.idle: '2021-06-26T16:35:15.86306Z', shell.execute_reply: '2021-06-26T16:35:15.86222Z', shell.execute_reply.started: '2021-06-26T16:35:15.855101Z'}
## Let's fix that
train.title = train.title.apply(lambda x: x.strip())
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:35:15.864826Z', iopub.status.busy: '2021-06-26T16:35:15.864362Z', iopub.status.idle: '2021-06-26T16:35:15.872663Z', shell.execute_reply: '2021-06-26T16:35:15.871817Z', shell.execute_reply.started: '2021-06-26T16:35:15.864612Z'}
## We can also combile all three lines above for test set here
test['title'] = [i.split('.')[0].split(',')[1].strip() for i in test.Name]

## However it is important to be able to write readable code, and the line above is not so readable.
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:35:15.874489Z', iopub.status.busy: '2021-06-26T16:35:15.873918Z', iopub.status.idle: '2021-06-26T16:35:15.896665Z', shell.execute_reply: '2021-06-26T16:35:15.895832Z', shell.execute_reply.started: '2021-06-26T16:35:15.874258Z'}
## Let's replace some of the rare values with the keyword 'rare' and other word choice of our own.
## train Data
train["title"] = [i.replace('Ms', 'Miss') for i in train.title]
train["title"] = [i.replace('Mlle', 'Miss') for i in train.title]
train["title"] = [i.replace('Mme', 'Mrs') for i in train.title]
train["title"] = [i.replace('Dr', 'rare') for i in train.title]
train["title"] = [i.replace('Col', 'rare') for i in train.title]
train["title"] = [i.replace('Major', 'rare') for i in train.title]
train["title"] = [i.replace('Don', 'rare') for i in train.title]
train["title"] = [i.replace('Jonkheer', 'rare') for i in train.title]
train["title"] = [i.replace('Sir', 'rare') for i in train.title]
train["title"] = [i.replace('Lady', 'rare') for i in train.title]
train["title"] = [i.replace('Capt', 'rare') for i in train.title]
train["title"] = [i.replace('the Countess', 'rare') for i in train.title]
train["title"] = [i.replace('Rev', 'rare') for i in train.title]


## Now in programming there is a term called DRY(Don't repeat yourself), whenever we are repeating
## same code over and over again, there should be a light-bulb turning on in our head and make us think
## to code in a way that is not repeating or dull. Let's write a function to do exactly what we
## did in the code above, only not repeating and more interesting.
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:35:15.900031Z', iopub.status.busy: '2021-06-26T16:35:15.899771Z', iopub.status.idle: '2021-06-26T16:35:15.910036Z', shell.execute_reply: '2021-06-26T16:35:15.908929Z', shell.execute_reply.started: '2021-06-26T16:35:15.899989Z'}
## we are writing a function that can help us modify title column
def name_converted(feature):
    """
    This function helps modifying the title column
    """

    result = ''
    if feature in ['the Countess','Capt','Lady','Sir','Jonkheer','Don','Major','Col', 'Rev', 'Dona', 'Dr']:
        result = 'rare'
    elif feature in ['Ms', 'Mlle']:
        result = 'Miss'
    elif feature == 'Mme':
        result = 'Mrs'
    else:
        result = feature
    return result

test.title = test.title.map(name_converted)
train.title = train.title.map(name_converted)
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:35:15.912187Z', iopub.status.busy: '2021-06-26T16:35:15.911644Z', iopub.status.idle: '2021-06-26T16:35:15.923512Z', shell.execute_reply: '2021-06-26T16:35:15.922507Z', shell.execute_reply.started: '2021-06-26T16:35:15.912136Z'}
print(train.title.unique())
print(test.title.unique())
#
#
#
#
#
#
#
#| _cell_guid: 7083a7e7-d1d5-4cc1-ad67-c454b139f5f1
#| _kg_hide-input: true
#| _uuid: cdfd54429cb235dd3b73535518950b2e515e54f2
#| execution: {iopub.execute_input: '2021-06-26T16:35:15.925581Z', iopub.status.busy: '2021-06-26T16:35:15.925033Z', iopub.status.idle: '2021-06-26T16:35:15.933955Z', shell.execute_reply: '2021-06-26T16:35:15.933137Z', shell.execute_reply.started: '2021-06-26T16:35:15.925315Z'}
## Family_size seems like a good feature to create
train['family_size'] = train.SibSp + train.Parch+1
test['family_size'] = test.SibSp + test.Parch+1
#
#
#
#| _cell_guid: 3d471d07-7735-4aab-8b26-3f26e481dc49
#| _kg_hide-input: true
#| _uuid: 2e23467af7a2e85fcaa06b52b303daf2e5e44250
#| execution: {iopub.execute_input: '2021-06-26T16:35:15.935971Z', iopub.status.busy: '2021-06-26T16:35:15.935422Z', iopub.status.idle: '2021-06-26T16:35:15.942647Z', shell.execute_reply: '2021-06-26T16:35:15.941882Z', shell.execute_reply.started: '2021-06-26T16:35:15.935671Z'}
## bin the family size.
def family_group(size):
    """
    This funciton groups(loner, small, large) family based on family size
    """

    a = ''
    if (size <= 1):
        a = 'loner'
    elif (size <= 4):
        a = 'small'
    else:
        a = 'large'
    return a
#
#
#
#| _cell_guid: 82f3cf5a-7e8d-42c3-a06b-56e17e890358
#| _kg_hide-input: true
#| _uuid: 549239812f919f5348da08db4264632d2b21b587
#| execution: {iopub.execute_input: '2021-06-26T16:35:15.944511Z', iopub.status.busy: '2021-06-26T16:35:15.94417Z', iopub.status.idle: '2021-06-26T16:35:15.95416Z', shell.execute_reply: '2021-06-26T16:35:15.953395Z', shell.execute_reply.started: '2021-06-26T16:35:15.944448Z'}
## apply the family_group function in family_size
train['family_group'] = train['family_size'].map(family_group)
test['family_group'] = test['family_size'].map(family_group)
#
#
#
#
#
#| _cell_guid: 298b28d6-75a7-4e49-b1c3-7755f1727327
#| _kg_hide-input: true
#| _uuid: 45315bb62f69e94e66109e7da06c6c5ade578398
#| execution: {iopub.execute_input: '2021-06-26T16:35:15.956031Z', iopub.status.busy: '2021-06-26T16:35:15.955569Z', iopub.status.idle: '2021-06-26T16:35:15.964779Z', shell.execute_reply: '2021-06-26T16:35:15.963853Z', shell.execute_reply.started: '2021-06-26T16:35:15.955855Z'}
train['is_alone'] = [1 if i<2 else 0 for i in train.family_size]
test['is_alone'] = [1 if i<2 else 0 for i in test.family_size]
#
#
#
#
#
#| _cell_guid: 352c794d-728d-44de-9160-25da7abe0c06
#| _kg_hide-input: true
#| _uuid: 5b99e1f7d7757f11e6dd6dbc627f3bd6e2fbd874
#| execution: {iopub.execute_input: '2021-06-26T16:35:15.966936Z', iopub.status.busy: '2021-06-26T16:35:15.9664Z', iopub.status.idle: '2021-06-26T16:35:15.97799Z', shell.execute_reply: '2021-06-26T16:35:15.976969Z', shell.execute_reply.started: '2021-06-26T16:35:15.966816Z'}
train.Ticket.value_counts().sample(10)
#
#
#
#
#
#| _kg_hide-input: true
#| _uuid: d23d451982f0cbe44976c2eacafb726d816e9195
#| execution: {iopub.execute_input: '2021-06-26T16:35:15.979613Z', iopub.status.busy: '2021-06-26T16:35:15.979155Z', iopub.status.idle: '2021-06-26T16:35:15.989456Z', shell.execute_reply: '2021-06-26T16:35:15.988913Z', shell.execute_reply.started: '2021-06-26T16:35:15.97941Z'}
train.drop(['Ticket'], axis=1, inplace=True)

test.drop(['Ticket'], axis=1, inplace=True)
#
#
#
#
#
#| _cell_guid: adaa30fe-cb0f-4666-bf95-505f1dcce188
#| _kg_hide-input: true
#| _uuid: 9374a6357551a7551e71731d72f5ceb3144856df
#| execution: {iopub.execute_input: '2021-06-26T16:35:15.991841Z', iopub.status.busy: '2021-06-26T16:35:15.991313Z', iopub.status.idle: '2021-06-26T16:35:15.999545Z', shell.execute_reply: '2021-06-26T16:35:15.998734Z', shell.execute_reply.started: '2021-06-26T16:35:15.991562Z'}
## Calculating fare based on family size.
train['calculated_fare'] = train.Fare/train.family_size
test['calculated_fare'] = test.Fare/test.family_size
#
#
#
#
#
#
#
#| _cell_guid: 8c33b78c-14cb-4cc2-af0f-65079a741570
#| _kg_hide-input: true
#| _uuid: 35685a6ca28651eab389c4673c21da2ea5ba4187
#| execution: {iopub.execute_input: '2021-06-26T16:35:16.001667Z', iopub.status.busy: '2021-06-26T16:35:16.001088Z', iopub.status.idle: '2021-06-26T16:35:16.012304Z', shell.execute_reply: '2021-06-26T16:35:16.011542Z', shell.execute_reply.started: '2021-06-26T16:35:16.00135Z'}
def fare_group(fare):
    """
    This function creates a fare group based on the fare provided
    """

    a= ''
    if fare <= 4:
        a = 'Very_low'
    elif fare <= 10:
        a = 'low'
    elif fare <= 20:
        a = 'mid'
    elif fare <= 45:
        a = 'high'
    else:
        a = "very_high"
    return a

train['fare_group'] = train['calculated_fare'].map(fare_group)
test['fare_group'] = test['calculated_fare'].map(fare_group)

#train['fare_group'] = pd.cut(train['calculated_fare'], bins = 4, labels=groups)
#
#
#
#
#
#
#
#
#
#| _uuid: dadea67801cf5b56a882aa96bb874a4afa0e0bec
#| execution: {iopub.execute_input: '2021-06-26T16:35:16.014434Z', iopub.status.busy: '2021-06-26T16:35:16.013951Z', iopub.status.idle: '2021-06-26T16:35:16.025524Z', shell.execute_reply: '2021-06-26T16:35:16.024631Z', shell.execute_reply.started: '2021-06-26T16:35:16.014266Z'}
train.drop(['PassengerId'], axis=1, inplace=True)

test.drop(['PassengerId'], axis=1, inplace=True)
#
#
#
#
#
#
#
#
#
#| _cell_guid: 9243ac8c-be44-46d0-a0ca-ee5f19b89bd4
#| _kg_hide-input: true
#| _uuid: 7b8db3930fb1bfb91db16686223dfc6d8e77744d
#| execution: {iopub.execute_input: '2021-06-26T16:35:16.027132Z', iopub.status.busy: '2021-06-26T16:35:16.026701Z', iopub.status.idle: '2021-06-26T16:35:16.059319Z', shell.execute_reply: '2021-06-26T16:35:16.058745Z', shell.execute_reply.started: '2021-06-26T16:35:16.027081Z'}

train = pd.get_dummies(train, columns=['title',"Pclass", 'Cabin','Embarked','nLength_group', 'family_group', 'fare_group'], drop_first=False)
test = pd.get_dummies(test, columns=['title',"Pclass",'Cabin','Embarked','nLength_group', 'family_group', 'fare_group'], drop_first=False)
train.drop(['family_size','Name', 'Fare','name_length'], axis=1, inplace=True)
test.drop(['Name','family_size',"Fare",'name_length'], axis=1, inplace=True)
#
#
#
#
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:35:16.061141Z', iopub.status.busy: '2021-06-26T16:35:16.060714Z', iopub.status.idle: '2021-06-26T16:35:16.084728Z', shell.execute_reply: '2021-06-26T16:35:16.083793Z', shell.execute_reply.started: '2021-06-26T16:35:16.060961Z'}
train.head()
#
#
#
#| _kg_hide-input: true
#| _uuid: 9597c320c3db4db5e5c28980a28abaae7281bc61
#| execution: {iopub.execute_input: '2021-06-26T16:35:16.086463Z', iopub.status.busy: '2021-06-26T16:35:16.086001Z', iopub.status.idle: '2021-06-26T16:35:16.096908Z', shell.execute_reply: '2021-06-26T16:35:16.095838Z', shell.execute_reply.started: '2021-06-26T16:35:16.086235Z'}
## rearranging the columns so that I can easily use the dataframe to predict the missing age values.
train = pd.concat([train[["Survived", "Age", "Sex","SibSp","Parch"]], train.loc[:,"is_alone":]], axis=1)
test = pd.concat([test[["Age", "Sex"]], test.loc[:,"SibSp":]], axis=1)
#
#
#
#| _kg_hide-input: true
#| _uuid: 91662e7b63c2361fdcf3215f130b3895154ad92d
#| execution: {iopub.execute_input: '2021-06-26T16:35:16.098683Z', iopub.status.busy: '2021-06-26T16:35:16.098263Z', iopub.status.idle: '2021-06-26T16:35:22.704889Z', shell.execute_reply: '2021-06-26T16:35:22.704165Z', shell.execute_reply.started: '2021-06-26T16:35:16.098504Z'}
## Importing RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

## writing a function that takes a dataframe with missing values and outputs it by filling the missing values.
def completing_age(df):
    ## gettting all the features except survived
    age_df = df.loc[:,"Age":]

    temp_train = age_df.loc[age_df.Age.notnull()] ## df with age values
    temp_test = age_df.loc[age_df.Age.isnull()] ## df without age values

    y = temp_train.Age.values ## setting target variables(age) in y
    x = temp_train.loc[:, "Sex":].values

    rfr = RandomForestRegressor(n_estimators=1500, n_jobs=-1)
    rfr.fit(x, y)

    predicted_age = rfr.predict(temp_test.loc[:, "Sex":])

    df.loc[df.Age.isnull(), "Age"] = predicted_age


    return df

## Implementing the completing_age function in both train and test dataset.
completing_age(train)
completing_age(test);
#
#
#
#
#
#| _kg_hide-input: true
#| _uuid: 8fc55e4670061d46dab3cc6585b3cc71eb996868
#| execution: {iopub.execute_input: '2021-06-26T16:35:22.708567Z', iopub.status.busy: '2021-06-26T16:35:22.708283Z', iopub.status.idle: '2021-06-26T16:35:23.194075Z', shell.execute_reply: '2021-06-26T16:35:23.193419Z', shell.execute_reply.started: '2021-06-26T16:35:22.708515Z'}
## Let's look at the his
plt.subplots(figsize = (22,10),)
sns.distplot(train.Age, bins = 100, kde = True, rug = False, norm_hist=False);
#
#
#
#
#
#
#
#| _cell_guid: 3140c968-6755-42ec-aa70-d30c0acede1e
#| _kg_hide-input: true
#| _uuid: c3bd77bb4d9d5411aa696a605be127db181d2a67
#| execution: {iopub.execute_input: '2021-06-26T16:35:23.196215Z', iopub.status.busy: '2021-06-26T16:35:23.195696Z', iopub.status.idle: '2021-06-26T16:35:23.219708Z', shell.execute_reply: '2021-06-26T16:35:23.218664Z', shell.execute_reply.started: '2021-06-26T16:35:23.195943Z'}
## create bins for age
def age_group_fun(age):
    """
    This function creates a bin for age
    """
    a = ''
    if age <= 1:
        a = 'infant'
    elif age <= 4:
        a = 'toddler'
    elif age <= 13:
        a = 'child'
    elif age <= 18:
        a = 'teenager'
    elif age <= 35:
        a = 'Young_Adult'
    elif age <= 45:
        a = 'adult'
    elif age <= 55:
        a = 'middle_aged'
    elif age <= 65:
        a = 'senior_citizen'
    else:
        a = 'old'
    return a

## Applying "age_group_fun" function to the "Age" column.
train['age_group'] = train['Age'].map(age_group_fun)
test['age_group'] = test['Age'].map(age_group_fun)

## Creating dummies for "age_group" feature.
train = pd.get_dummies(train,columns=['age_group'], drop_first=True)
test = pd.get_dummies(test,columns=['age_group'], drop_first=True);
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| _cell_guid: dcb0934f-8e3f-40b6-859e-abf70b0b074e
#| _kg_hide-input: true
#| _uuid: 607db6be6dfacc7385e5adcc0feeee28c50c99c5
#| execution: {iopub.execute_input: '2021-06-26T16:35:23.221875Z', iopub.status.busy: '2021-06-26T16:35:23.221297Z', iopub.status.idle: '2021-06-26T16:35:23.229845Z', shell.execute_reply: '2021-06-26T16:35:23.228853Z', shell.execute_reply.started: '2021-06-26T16:35:23.221578Z'}
# separating our independent and dependent variable
X = train.drop(['Survived'], axis = 1)
y = train["Survived"]
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| _cell_guid: 348a5be2-5f4f-4c98-93a3-7352b6060ef4
#| _kg_hide-input: true
#| _uuid: 41b70e57f8e03da9910c20af89a9fa4a2aaea85b
#| execution: {iopub.execute_input: '2021-06-26T16:35:23.231964Z', iopub.status.busy: '2021-06-26T16:35:23.23135Z', iopub.status.idle: '2021-06-26T16:35:23.240022Z', shell.execute_reply: '2021-06-26T16:35:23.239414Z', shell.execute_reply.started: '2021-06-26T16:35:23.231633Z'}
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = .33, random_state=0)
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:35:23.242734Z', iopub.status.busy: '2021-06-26T16:35:23.24208Z', iopub.status.idle: '2021-06-26T16:35:23.250654Z', shell.execute_reply: '2021-06-26T16:35:23.249893Z', shell.execute_reply.started: '2021-06-26T16:35:23.242373Z'}
len(X_train)
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:35:23.260997Z', iopub.status.busy: '2021-06-26T16:35:23.260779Z', iopub.status.idle: '2021-06-26T16:35:23.265643Z', shell.execute_reply: '2021-06-26T16:35:23.264688Z', shell.execute_reply.started: '2021-06-26T16:35:23.260954Z'}
len(X_test)
#
#
#
#
#
#
#
#
#
#
#
#| _kg_hide-input: true
#| _uuid: d788baa4b88106afe5b30c769a6c85a1d67a5d6c
#| execution: {iopub.execute_input: '2021-06-26T16:35:23.26761Z', iopub.status.busy: '2021-06-26T16:35:23.267136Z', iopub.status.idle: '2021-06-26T16:35:23.295264Z', shell.execute_reply: '2021-06-26T16:35:23.294322Z', shell.execute_reply.started: '2021-06-26T16:35:23.267383Z'}
train.sample(5)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| _kg_hide-input: true
#| _uuid: c4011a767b1d846f2866b4573d1d6d116afe8427
#| execution: {iopub.execute_input: '2021-06-26T16:35:23.297022Z', iopub.status.busy: '2021-06-26T16:35:23.296548Z', iopub.status.idle: '2021-06-26T16:35:23.319251Z', shell.execute_reply: '2021-06-26T16:35:23.318338Z', shell.execute_reply.started: '2021-06-26T16:35:23.296792Z'}
headers = X_train.columns

X_train.head()
#
#
#
#| _cell_guid: 5c89c54b-7f5a-4e31-9e8f-58726cef5eab
#| _kg_hide-input: true
#| _uuid: 182b849ba7f2b311e919cdbf83970b97736e9d98
#| execution: {iopub.execute_input: '2021-06-26T16:35:23.320979Z', iopub.status.busy: '2021-06-26T16:35:23.320476Z', iopub.status.idle: '2021-06-26T16:35:23.331478Z', shell.execute_reply: '2021-06-26T16:35:23.33067Z', shell.execute_reply.started: '2021-06-26T16:35:23.320738Z'}
# Feature Scaling
## We will be using standardscaler to transform
from sklearn.preprocessing import StandardScaler
st_scale = StandardScaler()

## transforming "train_x"
X_train = st_scale.fit_transform(X_train)
## transforming "test_x"
X_test = st_scale.transform(X_test)

## transforming "The testset"
#test = st_scale.transform(test)
#
#
#
#
#
#| _kg_hide-input: true
#| _uuid: fc6f031833ac9e2734aa7b3a2373b667679c6b2f
#| execution: {iopub.execute_input: '2021-06-26T16:35:23.333531Z', iopub.status.busy: '2021-06-26T16:35:23.333111Z', iopub.status.idle: '2021-06-26T16:35:23.359161Z', shell.execute_reply: '2021-06-26T16:35:23.358554Z', shell.execute_reply.started: '2021-06-26T16:35:23.333347Z'}
pd.DataFrame(X_train, columns=headers).head()
#
#
#
#
#
#
#
#
#
#
#
#
#
#| _cell_guid: 0c8b0c41-6738-4689-85b0-b83a16e46ab9
#| _uuid: 09140be1a71e37b441a16951a82747462b767e6e
#| execution: {iopub.execute_input: '2021-06-26T16:35:23.361067Z', iopub.status.busy: '2021-06-26T16:35:23.360637Z', iopub.status.idle: '2021-06-26T16:35:23.383762Z', shell.execute_reply: '2021-06-26T16:35:23.383049Z', shell.execute_reply.started: '2021-06-26T16:35:23.360889Z'}
# import LogisticRegression model in python.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score

## call on the model object
logreg = LogisticRegression(solver='liblinear',
                            penalty= 'l1',random_state = 42

                            )

## fit the model with "train_x" and "train_y"
logreg.fit(X_train,y_train)

## Once the model is trained we want to find out how well the model is performing, so we test the model.
## we use "X_test" portion of the data(this data was not used to fit the model) to predict model outcome.
y_pred = logreg.predict(X_test)

## Once predicted we save that outcome in "y_pred" variable.
## Then we compare the predicted value( "y_pred") and actual value("test_y") to see how well our model is performing.
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:35:23.385843Z', iopub.status.busy: '2021-06-26T16:35:23.385341Z', iopub.status.idle: '2021-06-26T16:35:23.399434Z', shell.execute_reply: '2021-06-26T16:35:23.398674Z', shell.execute_reply.started: '2021-06-26T16:35:23.385606Z'}
from sklearn.metrics import classification_report, confusion_matrix
# printing confision matrix
pd.DataFrame(confusion_matrix(y_test,y_pred),\
            columns=["Predicted Not-Survived", "Predicted Survived"],\
            index=["Not-Survived","Survived"] )
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:35:23.400652Z', iopub.status.busy: '2021-06-26T16:35:23.400403Z', iopub.status.idle: '2021-06-26T16:35:23.408635Z', shell.execute_reply: '2021-06-26T16:35:23.40776Z', shell.execute_reply.started: '2021-06-26T16:35:23.400604Z'}
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:35:23.410491Z', iopub.status.busy: '2021-06-26T16:35:23.410085Z', iopub.status.idle: '2021-06-26T16:35:23.418315Z', shell.execute_reply: '2021-06-26T16:35:23.417549Z', shell.execute_reply.started: '2021-06-26T16:35:23.410444Z'}
from sklearn.metrics import recall_score
recall_score(y_test, y_pred)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:35:23.4204Z', iopub.status.busy: '2021-06-26T16:35:23.419791Z', iopub.status.idle: '2021-06-26T16:35:23.429679Z', shell.execute_reply: '2021-06-26T16:35:23.42864Z', shell.execute_reply.started: '2021-06-26T16:35:23.420242Z'}
from sklearn.metrics import precision_score
precision_score(y_test, y_pred)
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:35:23.431682Z', iopub.status.busy: '2021-06-26T16:35:23.431234Z', iopub.status.idle: '2021-06-26T16:35:23.44225Z', shell.execute_reply: '2021-06-26T16:35:23.441202Z', shell.execute_reply.started: '2021-06-26T16:35:23.43147Z'}
from sklearn.metrics import classification_report, balanced_accuracy_score
print(classification_report(y_test, y_pred))
#
#
#
#
#
#| _kg_hide-input: true
#| execution: {iopub.execute_input: '2021-06-26T16:35:23.444153Z', iopub.status.busy: '2021-06-26T16:35:23.443714Z', iopub.status.idle: '2021-06-26T16:35:23.873374Z', shell.execute_reply: '2021-06-26T16:35:23.869521Z', shell.execute_reply.started: '2021-06-26T16:35:23.444104Z'}
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

class_names = np.array(['not_survived','survived'])

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
#
#
#
#
#
#| _uuid: 1e71bc7c685b757b6920076527780674d6f619bc
#| execution: {iopub.execute_input: '2021-06-26T16:35:23.877891Z', iopub.status.busy: '2021-06-26T16:35:23.875713Z', iopub.status.idle: '2021-06-26T16:35:24.505751Z', shell.execute_reply: '2021-06-26T16:35:24.501314Z', shell.execute_reply.started: '2021-06-26T16:35:23.87783Z'}
from sklearn.metrics import roc_curve, auc
#plt.style.use('seaborn-pastel')
y_score = logreg.decision_function(X_test)

FPR, TPR, _ = roc_curve(y_test, y_score)
ROC_AUC = auc(FPR, TPR)
print (ROC_AUC)

plt.figure(figsize =[11,9])
plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)
plt.plot([0,1],[0,1], 'k--', linewidth = 4)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate', fontsize = 18)
plt.ylabel('True Positive Rate', fontsize = 18)
plt.title('ROC for Titanic survivors', fontsize= 18)
plt.show()
#
#
#
#| _uuid: 22f15e384372a1ece2f28cd9eced0c703a79598f
#| execution: {iopub.execute_input: '2021-06-26T16:35:24.50731Z', iopub.status.busy: '2021-06-26T16:35:24.506981Z', iopub.status.idle: '2021-06-26T16:35:24.8481Z', shell.execute_reply: '2021-06-26T16:35:24.846974Z', shell.execute_reply.started: '2021-06-26T16:35:24.507251Z'}
from sklearn.metrics import precision_recall_curve

y_score = logreg.decision_function(X_test)

precision, recall, _ = precision_recall_curve(y_test, y_score)
PR_AUC = auc(recall, precision)

plt.figure(figsize=[11,9])
plt.plot(recall, precision, label='PR curve (area = %0.2f)' % PR_AUC, linewidth=4)
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.title('Precision Recall Curve for Titanic survivors', fontsize=18)
plt.legend(loc="lower right")
plt.show()
#
#
#
#
#
#
#
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:35:24.855506Z', iopub.status.busy: '2021-06-26T16:35:24.853028Z', iopub.status.idle: '2021-06-26T16:35:24.862513Z', shell.execute_reply: '2021-06-26T16:35:24.861421Z', shell.execute_reply.started: '2021-06-26T16:35:24.853368Z'}
sc = st_scale
#
#
#
#| _uuid: 17791284c3e88236de2daa112422cde8ddcb0641
#| execution: {iopub.execute_input: '2021-06-26T16:35:24.868704Z', iopub.status.busy: '2021-06-26T16:35:24.86826Z', iopub.status.idle: '2021-06-26T16:35:25.014634Z', shell.execute_reply: '2021-06-26T16:35:25.013771Z', shell.execute_reply.started: '2021-06-26T16:35:24.86853Z'}
#| scrolled: true
## Using StratifiedShuffleSplit
## We can use KFold, StratifiedShuffleSplit, StratiriedKFold or ShuffleSplit, They are all close cousins. look at sklearn userguide for more info.
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%
## Using standard scale for the whole dataset.

## saving the feature names for decision tree display
column_names = X.columns

X = sc.fit_transform(X)
accuracies = cross_val_score(LogisticRegression(solver='liblinear'), X,y, cv  = cv)
print ("Cross-Validation accuracy scores:{}".format(accuracies))
print ("Mean Cross-Validation accuracy score: {}".format(round(accuracies.mean(),5)))
#
#
#
#
#
#
#
#
#
#
#
#
#| _cell_guid: 0620523c-b33b-4302-8a1c-4b6759ffa5fa
#| _uuid: 36a379a00a31dd161be1723f65490990294fe13d
#| execution: {iopub.execute_input: '2021-06-26T16:35:25.021234Z', iopub.status.busy: '2021-06-26T16:35:25.018883Z', iopub.status.idle: '2021-06-26T16:35:40.193433Z', shell.execute_reply: '2021-06-26T16:35:40.192566Z', shell.execute_reply.started: '2021-06-26T16:35:25.021181Z'}
from sklearn.model_selection import GridSearchCV, StratifiedKFold
## C_vals is the alpla value of lasso and ridge regression(as alpha increases the model complexity decreases,)
## remember effective alpha scores are 0<alpha<infinity
C_vals = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,16.5,17,17.5,18]
## Choosing penalties(Lasso(l1) or Ridge(l2))
penalties = ['l1','l2']
## Choose a cross validation strategy.
cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25)

## setting param for param_grid in GridSearchCV.
param = {'penalty': penalties, 'C': C_vals}

logreg = LogisticRegression(solver='liblinear')
## Calling on GridSearchCV object.
grid = GridSearchCV(estimator=LogisticRegression(),
                           param_grid = param,
                           scoring = 'accuracy',
                            n_jobs =-1,
                           cv = cv
                          )
## Fitting the model
grid.fit(X, y)
#
#
#
#| _cell_guid: 1fa35072-87c4-4f47-86ab-dda03d4b7b15
#| _uuid: 4c6650e39550527b271ddf733dcfe5221bcd5c98
#| execution: {iopub.execute_input: '2021-06-26T16:35:40.195216Z', iopub.status.busy: '2021-06-26T16:35:40.194925Z', iopub.status.idle: '2021-06-26T16:35:40.201259Z', shell.execute_reply: '2021-06-26T16:35:40.200225Z', shell.execute_reply.started: '2021-06-26T16:35:40.19517Z'}
## Getting the best of everything.
print (grid.best_score_)
print (grid.best_params_)
print(grid.best_estimator_)

#
#
#
#
#
#| _uuid: ba53f6b3610821dc820936dde7b7803a54d20f5a
#| execution: {iopub.execute_input: '2021-06-26T16:35:40.204086Z', iopub.status.busy: '2021-06-26T16:35:40.203576Z', iopub.status.idle: '2021-06-26T16:35:40.214041Z', shell.execute_reply: '2021-06-26T16:35:40.212929Z', shell.execute_reply.started: '2021-06-26T16:35:40.20393Z'}
### Using the best parameters from the grid-search.
logreg_grid = grid.best_estimator_
logreg_grid.score(X,y)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| _uuid: 953bc2c18b5fd93bcd51a42cc04a0539d86d5bac
#| execution: {iopub.execute_input: '2021-06-26T16:35:40.216328Z', iopub.status.busy: '2021-06-26T16:35:40.215853Z', iopub.status.idle: '2021-06-26T16:35:40.416985Z', shell.execute_reply: '2021-06-26T16:35:40.416038Z', shell.execute_reply.started: '2021-06-26T16:35:40.216141Z'}
## Importing the model.
from sklearn.neighbors import KNeighborsClassifier
## calling on the model oject.
knn = KNeighborsClassifier(metric='minkowski', p=2)
## knn classifier works by doing euclidian distance


## doing 10 fold staratified-shuffle-split cross validation
cv = StratifiedShuffleSplit(n_splits=10, test_size=.25, random_state=2)

accuracies = cross_val_score(knn, X,y, cv = cv, scoring='accuracy')
print ("Cross-Validation accuracy scores:{}".format(accuracies))
print ("Mean Cross-Validation accuracy score: {}".format(round(accuracies.mean(),3)))
#
#
#
#
#
#| _uuid: 9c0f44165e08f63ae5436180c5a7182e6db5c63f
#| execution: {iopub.execute_input: '2021-06-26T16:35:40.418857Z', iopub.status.busy: '2021-06-26T16:35:40.418419Z', iopub.status.idle: '2021-06-26T16:35:46.541601Z', shell.execute_reply: '2021-06-26T16:35:46.540815Z', shell.execute_reply.started: '2021-06-26T16:35:40.418687Z'}
## Search for an optimal value of k for KNN.
k_range = range(1,31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X,y, cv = cv, scoring = 'accuracy')
    k_scores.append(scores.mean())
print("Accuracy scores are: {}\n".format(k_scores))
print ("Mean accuracy score: {}".format(np.mean(k_scores)))
#
#
#
#| _uuid: e123680b431ba99d399fa8205c32bcfdc7cabd81
#| execution: {iopub.execute_input: '2021-06-26T16:35:46.543234Z', iopub.status.busy: '2021-06-26T16:35:46.542789Z', iopub.status.idle: '2021-06-26T16:35:46.685143Z', shell.execute_reply: '2021-06-26T16:35:46.684141Z', shell.execute_reply.started: '2021-06-26T16:35:46.543184Z'}
from matplotlib import pyplot as plt
plt.plot(k_range, k_scores)
#
#
#
#
#
#| _uuid: 507e2a7cdb28a47be45ed247f1343c123a6b592b
#| execution: {iopub.execute_input: '2021-06-26T16:35:46.687026Z', iopub.status.busy: '2021-06-26T16:35:46.686671Z', iopub.status.idle: '2021-06-26T16:35:55.465245Z', shell.execute_reply: '2021-06-26T16:35:55.464452Z', shell.execute_reply.started: '2021-06-26T16:35:46.686956Z'}
from sklearn.model_selection import GridSearchCV
## trying out multiple values for k
k_range = range(1,31)
##
weights_options=['uniform','distance']
#
param = {'n_neighbors':k_range, 'weights':weights_options}
## Using startifiedShufflesplit.
cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)
# estimator = knn, param_grid = param, n_jobs = -1 to instruct scikit learn to use all available processors.
grid = GridSearchCV(KNeighborsClassifier(), param,cv=cv,verbose = False, n_jobs=-1)
## Fitting the model.
grid.fit(X,y)
#
#
#
#| _uuid: c710770daa6cf327dcc28e18b3ed180fabecd49b
#| execution: {iopub.execute_input: '2021-06-26T16:35:55.466929Z', iopub.status.busy: '2021-06-26T16:35:55.466654Z', iopub.status.idle: '2021-06-26T16:35:55.475348Z', shell.execute_reply: '2021-06-26T16:35:55.474575Z', shell.execute_reply.started: '2021-06-26T16:35:55.466883Z'}
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
#
#
#
#
#
#| _uuid: dd1fbf223c4ec9db65dde4924e2827e46029da1a
#| execution: {iopub.execute_input: '2021-06-26T16:35:55.477181Z', iopub.status.busy: '2021-06-26T16:35:55.476629Z', iopub.status.idle: '2021-06-26T16:35:55.555736Z', shell.execute_reply: '2021-06-26T16:35:55.554788Z', shell.execute_reply.started: '2021-06-26T16:35:55.476983Z'}
### Using the best parameters from the grid-search.
knn_grid= grid.best_estimator_
knn_grid.score(X,y)
#
#
#
#
#
#
#
#| _uuid: e159b267a57d7519fc0ee8b3d1e95b841d3daf60
#| execution: {iopub.execute_input: '2021-06-26T16:35:55.557501Z', iopub.status.busy: '2021-06-26T16:35:55.557097Z', iopub.status.idle: '2021-06-26T16:36:02.332003Z', shell.execute_reply: '2021-06-26T16:36:02.331364Z', shell.execute_reply.started: '2021-06-26T16:35:55.557338Z'}
from sklearn.model_selection import RandomizedSearchCV
## trying out multiple values for k
k_range = range(1,31)
##
weights_options=['uniform','distance']
#
param = {'n_neighbors':k_range, 'weights':weights_options}
## Using startifiedShufflesplit.
cv = StratifiedShuffleSplit(n_splits=10, test_size=.30)
# estimator = knn, param_grid = param, n_jobs = -1 to instruct scikit learn to use all available processors.
## for RandomizedSearchCV,
grid = RandomizedSearchCV(KNeighborsClassifier(), param,cv=cv,verbose = False, n_jobs=-1, n_iter=40)
## Fitting the model.
grid.fit(X,y)
#
#
#
#| _uuid: c58492525dd18659ef9f9c774ee7601a55e96f36
#| execution: {iopub.execute_input: '2021-06-26T16:36:02.333632Z', iopub.status.busy: '2021-06-26T16:36:02.333341Z', iopub.status.idle: '2021-06-26T16:36:02.340211Z', shell.execute_reply: '2021-06-26T16:36:02.338113Z', shell.execute_reply.started: '2021-06-26T16:36:02.333572Z'}
print (grid.best_score_)
print (grid.best_params_)
print(grid.best_estimator_)
#
#
#
#| _uuid: 6fb31588585d50de773ba0db6c378363841a5313
#| execution: {iopub.execute_input: '2021-06-26T16:36:02.343117Z', iopub.status.busy: '2021-06-26T16:36:02.34256Z', iopub.status.idle: '2021-06-26T16:36:02.420683Z', shell.execute_reply: '2021-06-26T16:36:02.419712Z', shell.execute_reply.started: '2021-06-26T16:36:02.342922Z'}
### Using the best parameters from the grid-search.
knn_ran_grid = grid.best_estimator_
knn_ran_grid.score(X,y)
#
#
#
#
#
#
#
#
#
#| _uuid: 8b2435030dbef1303bfc2864d227f5918f359330
#| execution: {iopub.execute_input: '2021-06-26T16:36:02.422487Z', iopub.status.busy: '2021-06-26T16:36:02.421997Z', iopub.status.idle: '2021-06-26T16:36:02.433216Z', shell.execute_reply: '2021-06-26T16:36:02.43234Z', shell.execute_reply.started: '2021-06-26T16:36:02.422237Z'}
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(X, y)
y_pred = gaussian.predict(X_test)
gaussian_accy = round(accuracy_score(y_pred, y_test), 3)
print(gaussian_accy)
#
#
#
#
#
#
#
#
#
#| _uuid: 56895672215b0b6365c6aaa10e446216ef635f53
#| execution: {iopub.execute_input: '2021-06-26T16:36:02.435838Z', iopub.status.busy: '2021-06-26T16:36:02.435282Z', iopub.status.idle: '2021-06-26T16:37:25.882123Z', shell.execute_reply: '2021-06-26T16:37:25.881483Z', shell.execute_reply.started: '2021-06-26T16:36:02.435553Z'}
from sklearn.svm import SVC
Cs = [0.001, 0.01, 0.1, 1,1.5,2,2.5,3,4,5, 10] ## penalty parameter C for the error term.
gammas = [0.0001,0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas}
cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)
grid_search = GridSearchCV(SVC(kernel = 'rbf', probability=True), param_grid, cv=cv) ## 'rbf' stands for gaussian kernel
grid_search.fit(X,y)
#
#
#
#| _uuid: 4108264ea5d18e3d3fa38a30584a032c734d6d49
#| execution: {iopub.execute_input: '2021-06-26T16:37:25.8839Z', iopub.status.busy: '2021-06-26T16:37:25.883609Z', iopub.status.idle: '2021-06-26T16:37:25.890029Z', shell.execute_reply: '2021-06-26T16:37:25.889244Z', shell.execute_reply.started: '2021-06-26T16:37:25.883852Z'}
print(grid_search.best_score_)
print(grid_search.best_params_)
print(grid_search.best_estimator_)
#
#
#
#| _uuid: db18a3b5475f03b21a039e31e4962c43f7caffdc
#| execution: {iopub.execute_input: '2021-06-26T16:37:25.892123Z', iopub.status.busy: '2021-06-26T16:37:25.891542Z', iopub.status.idle: '2021-06-26T16:37:25.934216Z', shell.execute_reply: '2021-06-26T16:37:25.933352Z', shell.execute_reply.started: '2021-06-26T16:37:25.892073Z'}
# using the best found hyper paremeters to get the score.
svm_grid = grid_search.best_estimator_
svm_grid.score(X,y)
#
#
#
#
#
#
#
#| _cell_guid: 38c90de9-d2e9-4341-a378-a854762d8be2
#| _uuid: 18efb62b713591d1512010536ff10d9f6a91ec11
#| execution: {iopub.execute_input: '2021-06-26T16:37:25.936111Z', iopub.status.busy: '2021-06-26T16:37:25.935654Z', iopub.status.idle: '2021-06-26T16:37:57.983942Z', shell.execute_reply: '2021-06-26T16:37:57.983035Z', shell.execute_reply.started: '2021-06-26T16:37:25.935918Z'}
from sklearn.tree import DecisionTreeClassifier
max_depth = range(1,30)
max_feature = [21,22,23,24,25,26,28,29,30,'auto']
criterion=["entropy", "gini"]

param = {'max_depth':max_depth,
         'max_features':max_feature,
         'criterion': criterion}
grid = GridSearchCV(DecisionTreeClassifier(),
                                param_grid = param,
                                 verbose=False,
                                 cv=StratifiedKFold(n_splits=20, random_state=15, shuffle=True),
                                n_jobs = -1)
grid.fit(X, y)
#
#
#
#| _cell_guid: b2222e4e-f5f2-4601-b95f-506d7811610a
#| _uuid: b0fb5055e6b4a7fb69ef44f669c4df693ce46212
#| execution: {iopub.execute_input: '2021-06-26T16:37:57.988346Z', iopub.status.busy: '2021-06-26T16:37:57.988045Z', iopub.status.idle: '2021-06-26T16:37:57.994617Z', shell.execute_reply: '2021-06-26T16:37:57.993662Z', shell.execute_reply.started: '2021-06-26T16:37:57.988287Z'}
#| scrolled: true
print( grid.best_params_)
print (grid.best_score_)
print (grid.best_estimator_)
#
#
#
#| _cell_guid: d731079a-31b4-429a-8445-48597bb2639d
#| _uuid: 76c26437d374442826ef140574c5c4880ae1e853
#| execution: {iopub.execute_input: '2021-06-26T16:37:57.996876Z', iopub.status.busy: '2021-06-26T16:37:57.996238Z', iopub.status.idle: '2021-06-26T16:37:58.010892Z', shell.execute_reply: '2021-06-26T16:37:58.010194Z', shell.execute_reply.started: '2021-06-26T16:37:57.996695Z'}
dectree_grid = grid.best_estimator_
## using the best found hyper paremeters to get the score.
dectree_grid.score(X,y)
#
#
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:37:58.013756Z', iopub.status.busy: '2021-06-26T16:37:58.01221Z', iopub.status.idle: '2021-06-26T16:37:58.034194Z', shell.execute_reply: '2021-06-26T16:37:58.033436Z', shell.execute_reply.started: '2021-06-26T16:37:58.013683Z'}
## feature importance
feature_importances = pd.DataFrame(dectree_grid.feature_importances_,
                                   index = column_names,
                                    columns=['importance'])
feature_importances.sort_values(by='importance', ascending=False).head(10)
#
#
#
#
#
#
#
#
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:37:58.040453Z', iopub.status.busy: '2021-06-26T16:37:58.038063Z', iopub.status.idle: '2021-06-26T16:39:53.557817Z', shell.execute_reply: '2021-06-26T16:39:53.556973Z', shell.execute_reply.started: '2021-06-26T16:37:58.040398Z'}
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
n_estimators = [140,145,150,155,160];
max_depth = range(1,10);
criterions = ['gini', 'entropy'];
cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)


parameters = {'n_estimators':n_estimators,
              'max_depth':max_depth,
              'criterion': criterions

        }
grid = GridSearchCV(estimator=RandomForestClassifier(max_features='auto'),
                                 param_grid=parameters,
                                 cv=cv,
                                 n_jobs = -1)
grid.fit(X,y)
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:39:53.559492Z', iopub.status.busy: '2021-06-26T16:39:53.559192Z', iopub.status.idle: '2021-06-26T16:39:53.567897Z', shell.execute_reply: '2021-06-26T16:39:53.56675Z', shell.execute_reply.started: '2021-06-26T16:39:53.559434Z'}
print (grid.best_score_)
print (grid.best_params_)
print (grid.best_estimator_)
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:39:53.570209Z', iopub.status.busy: '2021-06-26T16:39:53.56951Z', iopub.status.idle: '2021-06-26T16:39:53.600458Z', shell.execute_reply: '2021-06-26T16:39:53.599531Z', shell.execute_reply.started: '2021-06-26T16:39:53.569928Z'}
rf_grid = grid.best_estimator_
rf_grid.score(X,y)
#
#
#
#| _kg_hide-input: true
#| execution: {iopub.execute_input: '2021-06-26T16:39:53.602628Z', iopub.status.busy: '2021-06-26T16:39:53.602028Z', iopub.status.idle: '2021-06-26T16:39:53.613347Z', shell.execute_reply: '2021-06-26T16:39:53.612229Z', shell.execute_reply.started: '2021-06-26T16:39:53.602297Z'}
from sklearn.metrics import classification_report
# Print classification report for y_test
print(classification_report(y_test, y_pred, labels=rf_grid.classes_))
#
#
#
#
#
#| _kg_hide-input: true
#| execution: {iopub.execute_input: '2021-06-26T16:39:53.615537Z', iopub.status.busy: '2021-06-26T16:39:53.614947Z', iopub.status.idle: '2021-06-26T16:39:53.637392Z', shell.execute_reply: '2021-06-26T16:39:53.63647Z', shell.execute_reply.started: '2021-06-26T16:39:53.615192Z'}
## feature importance
feature_importances = pd.DataFrame(rf_grid.feature_importances_,
                                   index = column_names,
                                    columns=['importance'])
feature_importances.sort_values(by='importance', ascending=False).head(10)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:39:53.639198Z', iopub.status.busy: '2021-06-26T16:39:53.63871Z', iopub.status.idle: '2021-06-26T16:40:17.162923Z', shell.execute_reply: '2021-06-26T16:40:17.162277Z', shell.execute_reply.started: '2021-06-26T16:39:53.638945Z'}
from sklearn.ensemble import BaggingClassifier
n_estimators = [10,30,50,70,80,150,160, 170,175,180,185];
cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)

parameters = {'n_estimators':n_estimators,

        }
grid = GridSearchCV(BaggingClassifier(base_estimator= None, ## If None, then the base estimator is a decision tree.
                                      bootstrap_features=False),
                                 param_grid=parameters,
                                 cv=cv,
                                 n_jobs = -1)
grid.fit(X,y)
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:40:17.164621Z', iopub.status.busy: '2021-06-26T16:40:17.164322Z', iopub.status.idle: '2021-06-26T16:40:17.172911Z', shell.execute_reply: '2021-06-26T16:40:17.172302Z', shell.execute_reply.started: '2021-06-26T16:40:17.164559Z'}
print (grid.best_score_)
print (grid.best_params_)
print (grid.best_estimator_)
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:40:17.174968Z', iopub.status.busy: '2021-06-26T16:40:17.174466Z', iopub.status.idle: '2021-06-26T16:40:17.226122Z', shell.execute_reply: '2021-06-26T16:40:17.225161Z', shell.execute_reply.started: '2021-06-26T16:40:17.174765Z'}
bagging_grid = grid.best_estimator_
bagging_grid.score(X,y)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:40:17.227822Z', iopub.status.busy: '2021-06-26T16:40:17.227396Z', iopub.status.idle: '2021-06-26T16:41:28.311627Z', shell.execute_reply: '2021-06-26T16:41:28.311009Z', shell.execute_reply.started: '2021-06-26T16:40:17.227656Z'}
from sklearn.ensemble import AdaBoostClassifier
n_estimators = [100,140,145,150,160, 170,175,180,185];
cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)
learning_r = [0.1,1,0.01,0.5]

parameters = {'n_estimators':n_estimators,
              'learning_rate':learning_r

        }
grid = GridSearchCV(AdaBoostClassifier(base_estimator= None, ## If None, then the base estimator is a decision tree.
                                     ),
                                 param_grid=parameters,
                                 cv=cv,
                                 n_jobs = -1)
grid.fit(X,y)
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:41:28.313135Z', iopub.status.busy: '2021-06-26T16:41:28.31287Z', iopub.status.idle: '2021-06-26T16:41:28.318909Z', shell.execute_reply: '2021-06-26T16:41:28.318191Z', shell.execute_reply.started: '2021-06-26T16:41:28.313088Z'}
print (grid.best_score_)
print (grid.best_params_)
print (grid.best_estimator_)
#
#
#
#| execution: {iopub.execute_input: '2021-06-26T16:41:28.320845Z', iopub.status.busy: '2021-06-26T16:41:28.320267Z', iopub.status.idle: '2021-06-26T16:41:28.35912Z', shell.execute_reply: '2021-06-26T16:41:28.358535Z', shell.execute_reply.started: '2021-06-26T16:41:28.320797Z'}
adaBoost_grid = grid.best_estimator_
adaBoost_grid.score(X,y)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| _cell_guid: d32d6df9-b8e7-4637-bacc-2baec08547b8
#| _uuid: fd788c4f4cde834a1329f325f1f59e3f77c37e42
#| execution: {iopub.execute_input: '2021-06-26T16:41:28.360536Z', iopub.status.busy: '2021-06-26T16:41:28.360265Z', iopub.status.idle: '2021-06-26T16:41:28.521396Z', shell.execute_reply: '2021-06-26T16:41:28.520426Z', shell.execute_reply.started: '2021-06-26T16:41:28.360479Z'}
#| scrolled: true
# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gradient_boost = GradientBoostingClassifier()
gradient_boost.fit(X, y)
y_pred = gradient_boost.predict(X_test)
gradient_accy = round(accuracy_score(y_pred, y_test), 3)
print(gradient_accy)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| _cell_guid: 5d94cc5b-d8b7-40d3-b264-138539daabfa
#| _uuid: 9d96154d2267ea26a6682a73bd1850026eb1303b
#| execution: {iopub.execute_input: '2021-06-26T16:41:28.523177Z', iopub.status.busy: '2021-06-26T16:41:28.522724Z', iopub.status.idle: '2021-06-26T16:41:28.526955Z', shell.execute_reply: '2021-06-26T16:41:28.525945Z', shell.execute_reply.started: '2021-06-26T16:41:28.522964Z'}
# from xgboost import XGBClassifier
# XGBClassifier = XGBClassifier()
# XGBClassifier.fit(X, y)
# y_pred = XGBClassifier.predict(X_test)
# XGBClassifier_accy = round(accuracy_score(y_pred, y_test), 3)
# print(XGBClassifier_accy)
#
#
#
#
#
#
#
#
#
#| _cell_guid: 2e567e01-6b5f-4313-84af-cc378c3b709e
#| _uuid: c9b958e2488adf6f79401c677087e3250d63ac9b
#| execution: {iopub.execute_input: '2021-06-26T16:41:28.528841Z', iopub.status.busy: '2021-06-26T16:41:28.528382Z', iopub.status.idle: '2021-06-26T16:41:28.555697Z', shell.execute_reply: '2021-06-26T16:41:28.554889Z', shell.execute_reply.started: '2021-06-26T16:41:28.528664Z'}
from sklearn.ensemble import ExtraTreesClassifier
ExtraTreesClassifier = ExtraTreesClassifier()
ExtraTreesClassifier.fit(X, y)
y_pred = ExtraTreesClassifier.predict(X_test)
extraTree_accy = round(accuracy_score(y_pred, y_test), 3)
print(extraTree_accy)
#
#
#
#
#
#
#
#
#
#| _cell_guid: 23bd5744-e04d-49bb-9d70-7c2a518f76dd
#| _uuid: 57fc008eea2ce1c0b595f888a82ddeaee6ce2177
#| execution: {iopub.execute_input: '2021-06-26T16:41:28.557268Z', iopub.status.busy: '2021-06-26T16:41:28.556845Z', iopub.status.idle: '2021-06-26T16:41:28.863352Z', shell.execute_reply: '2021-06-26T16:41:28.862576Z', shell.execute_reply.started: '2021-06-26T16:41:28.557221Z'}
from sklearn.gaussian_process import GaussianProcessClassifier
GaussianProcessClassifier = GaussianProcessClassifier()
GaussianProcessClassifier.fit(X, y)
y_pred = GaussianProcessClassifier.predict(X_test)
gau_pro_accy = round(accuracy_score(y_pred, y_test), 3)
print(gau_pro_accy)
#
#
#
#
#
#
#
#
#
#| _cell_guid: ac208dd3-1045-47bb-9512-de5ecb5c81b0
#| _uuid: 821c74bbf404193219eb91fe53755d669f5a14d1
#| execution: {iopub.execute_input: '2021-06-26T16:41:28.865063Z', iopub.status.busy: '2021-06-26T16:41:28.86463Z', iopub.status.idle: '2021-06-26T16:41:30.314425Z', shell.execute_reply: '2021-06-26T16:41:30.313671Z', shell.execute_reply.started: '2021-06-26T16:41:28.865013Z'}
from sklearn.ensemble import VotingClassifier

voting_classifier = VotingClassifier(estimators=[
    ('lr_grid', logreg_grid),
    ('svc', svm_grid),
    ('random_forest', rf_grid),
    ('gradient_boosting', gradient_boost),
    ('decision_tree_grid',dectree_grid),
    ('knn_classifier', knn_grid),
#     ('XGB_Classifier', XGBClassifier),
    ('bagging_classifier', bagging_grid),
    ('adaBoost_classifier',adaBoost_grid),
    ('ExtraTrees_Classifier', ExtraTreesClassifier),
    ('gaussian_classifier',gaussian),
    ('gaussian_process_classifier', GaussianProcessClassifier)
],voting='hard')

#voting_classifier = voting_classifier.fit(train_x,train_y)
voting_classifier = voting_classifier.fit(X,y)
#
#
#
#| _cell_guid: 648ac6a6-2437-490a-bf76-1612a71126e8
#| _uuid: 518a02ae91cc91d618e476d1fc643cd3912ee5fb
#| execution: {iopub.execute_input: '2021-06-26T16:41:30.316454Z', iopub.status.busy: '2021-06-26T16:41:30.316008Z', iopub.status.idle: '2021-06-26T16:41:30.42114Z', shell.execute_reply: '2021-06-26T16:41:30.420152Z', shell.execute_reply.started: '2021-06-26T16:41:30.31627Z'}
y_pred = voting_classifier.predict(X_test)
voting_accy = round(accuracy_score(y_pred, y_test), 3)
print(voting_accy)
#
#
#
#| _cell_guid: 277534eb-7ec8-4359-a2f4-30f7f76611b8
#| _kg_hide-input: true
#| _uuid: 00a9b98fd4e230db427a63596a2747f05b1654c1
#| execution: {iopub.execute_input: '2021-06-26T16:41:30.422908Z', iopub.status.busy: '2021-06-26T16:41:30.422475Z', iopub.status.idle: '2021-06-26T16:41:30.426856Z', shell.execute_reply: '2021-06-26T16:41:30.425882Z', shell.execute_reply.started: '2021-06-26T16:41:30.422736Z'}
#models = pd.DataFrame({
#    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
#              'Random Forest', 'Naive Bayes',
#              'Decision Tree', 'Gradient Boosting Classifier', 'Voting Classifier', 'XGB Classifier','ExtraTrees Classifier','Bagging Classifier'],
#    'Score': [svc_accy, knn_accy, logreg_accy,
#              random_accy, gaussian_accy, dectree_accy,
#               gradient_accy, voting_accy, XGBClassifier_accy, extraTree_accy, bagging_accy]})
#models.sort_values(by='Score', ascending=False)
#
#
#
#
#
#
#
#
#
#| _uuid: eb0054822f296ba86aa6005b2a5e35fbc1aec88b
#| execution: {iopub.execute_input: '2021-06-26T16:41:30.429099Z', iopub.status.busy: '2021-06-26T16:41:30.42862Z', iopub.status.idle: '2021-06-26T16:41:30.646363Z', shell.execute_reply: '2021-06-26T16:41:30.645616Z', shell.execute_reply.started: '2021-06-26T16:41:30.428903Z'}
all_models = [logreg_grid,
              knn_grid,
              knn_ran_grid,
              svm_grid,
              dectree_grid,
              rf_grid,
              bagging_grid,
              adaBoost_grid,
              voting_classifier]

c = {}
for i in all_models:
    a = i.predict(X_test)
    b = accuracy_score(a, y_test)
    c[i] = b

#
#
#
#| _cell_guid: 51368e53-52e4-41cf-9cc9-af6164c9c6f5
#| _uuid: b947f168f6655c1c6eadaf53f3485d57c0cd74c7
#| execution: {iopub.execute_input: '2021-06-26T16:41:30.648318Z', iopub.status.busy: '2021-06-26T16:41:30.647987Z', iopub.status.idle: '2021-06-26T16:41:32.045557Z', shell.execute_reply: '2021-06-26T16:41:32.044733Z', shell.execute_reply.started: '2021-06-26T16:41:30.648259Z'}
test_prediction = (max(c, key=c.get)).predict(test)
submission = pd.DataFrame({
        "PassengerId": passengerid,
        "Survived": test_prediction
    })

submission.PassengerId = submission.PassengerId.astype(int)
submission.Survived = submission.Survived.astype(int)

submission.to_csv("titanic1_submission.csv", index=False)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
