import re

def MergeParameters(tac_fname, lmap_fname, new_tac_fname, new_lmap_fname):
    new_lmap_content = []
    positive_transition_cache = {}
    negative_transition_cache = {}
    threshold_transition_cache = {}
    gamma_transition_cache = {}
    emission_transition_cache = {}
    positive_initial_transition = re.compile(r"([0-9]+) p 0 H1=state([0-9]+) \| H0=state([0-9]+) +\+")
    positive_transition = re.compile(r"([0-9]+) p 0 H[0-9]+=state([0-9]+) \| H[0-9]+=state([0-9]+) +\+")
    negative_initial_transition = re.compile(r"([0-9]+) p 0 H1=state([0-9]+) \| H0=state([0-9]+) +\-")
    negative_transition = re.compile(r"([0-9]+) p 0 H[0-9]+=state([0-9]+) \| H[0-9]+=state([0-9]+) +\-")
    threshold_initial_transition = re.compile(r"([0-9]+) p 0 H1 \| H0=state([0-9]+) +Thres")
    threshold_transition = re.compile(r"([0-9]+) p 0 H[0-9]+ \| H[0-9]+=state([0-9]+) +Thres")
    gamma_initial_transition = re.compile(r"([0-9]+) p 0 H1 \| H0=state([0-9]+) +Gamma")
    gamma_transition = re.compile(r"([0-9]+) p 0 H[0-9]+ \| H[0-9]+=state([0-9]+) +Gamma")
    emission_initial_transition = re.compile(r"([0-9]+) p 0 E0=state([0-9]+) \| H0=state([0-9]+)")
    emission_transition = re.compile(r"([0-9]+) p 0 E[0-9]+=state([0-9]+) \| H[0-9]+=state([0-9]+)")
    literal_map = {}
    with open(lmap_fname, "r") as fp:
        for line in fp:
            line = line.strip()
            match = positive_initial_transition.match(line)
            if match:
                positive_transition_cache[(match.group(2), match.group(3))] = match.group(1)
                new_lmap_content.append("%s p 0 Hi=state%s | Hj=state%s +" % (match.group(1), match.group(2), match.group(3)))
                continue
            match = negative_initial_transition.match(line)
            if match:
                negative_transition_cache[(match.group(2), match.group(3))] = match.group(1)
                new_lmap_content.append("%s p 0 Hi=state%s | Hj=state%s -" % (match.group(1), match.group(2), match.group(3)))
                continue
            match = positive_transition.match(line)
            if match:
                literal_map[match.group(1)] = positive_transition_cache[(match.group(2), match.group(3))]
                continue
            match = negative_transition.match(line)
            if match:
                literal_map[match.group(1)] = negative_transition_cache[(match.group(2), match.group(3))]
                continue
            match = threshold_initial_transition.match(line)
            if match:
                threshold_transition_cache[match.group(2)] = match.group(1)
                new_lmap_content.append("%s p 0 Hi | Hj=state%s Thres" % (match.group(1), match.group(2)))
                continue
            match = threshold_transition.match(line)
            if match:
                literal_map[match.group(1)] = threshold_transition_cache[match.group(2)]
                continue
            match = gamma_initial_transition.match(line)
            if match:
                gamma_transition_cache[match.group(2)] = match.group(1)
                new_lmap_content.append("%s p 0 Hi | Hj=state%s Gamma" % (match.group(1), match.group(2)))
                continue
            match = gamma_transition.match(line)
            if match:
                literal_map[match.group(1)] = gamma_transition_cache[match.group(2)]
                continue;
            match = emission_initial_transition.match(line)
            if match:
                emission_transition_cache[(match.group(2), match.group(3))] = match.group(1)
                new_lmap_content.append("%s p 0 Ei=state%s | Hj=state%s" % (match.group(1), match.group(2), match.group(3)))
                continue
            match = emission_transition.match(line)
            if match:
                literal_map[match.group(1)] = emission_transition_cache[(match.group(2), match.group(3))]
                continue;
            # initial parameter
            new_lmap_content.append(line)
    new_tac_content = []
    literal_id_to_node_id = {}
    old_node_id_to_new_literal_id = {}
    with open(tac_fname, "r") as fp:
        for line in fp:
            line = line.strip()
            toks = line.split()
            if toks[1] == "L":
                literal_id_to_node_id[toks[2]] = toks[0]
                if toks[2] in literal_map:
                    old_node_id_to_new_literal_id[toks[0]] = literal_map[toks[2]]
                else:
                    new_tac_content.append(line)
            else:
                for i in range(3, len(toks)):
                    if toks[i] in old_node_id_to_new_literal_id:
                        replacing_literal_id = old_node_id_to_new_literal_id[toks[i]]
                        updated_node_id = literal_id_to_node_id[replacing_literal_id]
                        toks[i] = updated_node_id
                new_tac_content.append(" ".join(toks))
    with open(new_tac_fname, "w") as fp:
        fp.write("\n".join(new_tac_content))
    with open(new_lmap_fname, "w") as fp:
        fp.write("\n".join(new_lmap_content))

if __name__ == "__main__":
    import sys
    if len(sys.argv) <= 4:
        print ("Usage: <tac_fname> <lmap_fname> <new_tac_fname> <new_lmap_fname>")
        sys.exit(1)
    tac_fname = sys.argv[1]
    lmap_fname = sys.argv[2]
    new_tac_fname = sys.argv[3]
    new_lmap_fname = sys.argv[4]
    MergeParameters(tac_fname, lmap_fname, new_tac_fname, new_lmap_fname)
