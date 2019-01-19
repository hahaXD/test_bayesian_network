
def GenerateHiddenStateInfo(node_id, hidden_state_size, is_query, is_test):
    content = []
    content.append("node H%s" % node_id)
    content.append("{")
    content.append("states = (%s);"%(" ".join(["\"state%s\""%i for i in range(0, hidden_state_size)])))
    if is_query:
        content.append("diagnosistype = \"TARGET\";")
    else:
        content.append("diagnosistype = \"AUXILIARY\";")
    if node_id == 0 or not is_test:
        content.append("isdecisionvariable = \"false\";")
    else:
        content.append("isdecisionvariable = \"true\";")
    content.append("}")
    return content

def GenerateEmissionStateInfo(node_id):
    content = []
    content.append("node E%s" % node_id)
    content.append("{")
    content.append("states = (%s);"%(" ".join(["\"state%s\""%i for i in range(0, 2)])))
    content.append("diagnosistype = \"OBSERVATION\";")
    content.append("isdecisionvariable = \"false\";")
    content.append("}")
    return content


def GenerateHmmNet(chain_size, hidden_state_size, hmm_filename, is_test=True):
    total_content = []
    for i in range (0, chain_size):
        total_content += GenerateHiddenStateInfo(i, hidden_state_size, i == (chain_size-1), is_test)
        total_content += GenerateEmissionStateInfo(i)
    for i in range(0, chain_size):
        # Add transition potential
        if i != 0:
            total_content.append("potential ( H%s | H%s )" % (i, i-1))
        else:
            total_content.append("potential ( H0 | )")
        # Add emission potential
        total_content.append("potential ( E%s | H%s )" % (i,i))
    with open (hmm_filename, "w") as fp:
        fp.write("\n".join(total_content))

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print ("Usage <chain_size> <hidden_state_size> <output_filename>")
        sys.exit(1)
    chain_size = int(sys.argv[1])
    hidden_state_size = int(sys.argv[2])
    output_filename = sys.argv[3]
    GenerateHmmNet(chain_size, hidden_state_size, output_filename)

