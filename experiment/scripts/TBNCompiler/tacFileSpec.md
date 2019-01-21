Every TAC is stored in two files:*.tac and *.lmap

The tac file stores the structure of the TAC, and lmap file stores the tac's literals including parameters and indicators

**tac file**

Every line is like the following:

```
NodeID NodeType Specs
```

Specs will be set according to the NodeType

**NodeType:**

- L: Parameter node or Indicator node. Specs = [-]LiteralID, which helps to find the corresponding literal in lmap file. For any L representing an indicator node, presence of the negative sign means that the indicator corresponds to negative instantiation, absence of negative sign means a positive instantiation.
- *,+: Multiply and Add nodes. Specs = NumOfChildren ChildID1 ChildID2 ...
- ? : Testing node. Specs = 4 ParentProb Threshold ThetaPositive ThetaNegative
- Z: Normalizing node. Specs = NumOfChildren ChildID1 ChildID2 .... Result will be the value of node ChildID1 divided by the sum of all the children nodes'values.



**lmap file**

Every line is one of the following two:

```
LiteralID p Value Label

LiteralID i Symbol Label
```

Label should be ignored.



