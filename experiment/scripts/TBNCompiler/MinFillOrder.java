
import il2.inf.structure.EliminationOrders;
import il2.model.*;
import il2.util.IntSet;
import il2.util.*;

import java.util.ArrayList;
import java.io.File;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class MinFillOrder
{
	public static void main(String[] args)
	{
		MinFillOrder inf_engine = new MinFillOrder();
		Domain domain = new Domain();
		ArrayList<Index> subdomains = inf_engine.readBayesNet(domain);
		// obtain the subdomains of this network
		int[] query = new int[args.length];
		for (int i = 0; i < args.length; i++)
			query[i] = domain.index(args[i]);
		// obtain the id of query variables
		int[] mf_order_id = EliminationOrders.constrainedMinFill(subdomains, new IntSet(query)).order.toArray();
		// obtain a minfill elimination order of variables with query at the last
		String[] mf_order = new String[mf_order_id.length];
		for (int j = 0; j < mf_order_id.length; j++)
			mf_order[j] = domain.name(mf_order_id[j]);
		String output = String.join(",", mf_order);

		try
		{
			String class_path = System.getProperty("java.class.path").split(":")[0];
			String[] order_path = new String[] {class_path, "..", "tmp", "order.txt"};
			File order = new File(String.join(File.separator, order_path));
			order.createNewFile();
			FileWriter writer = new FileWriter(order, false);
			writer.write(output + "\n");
			writer.flush();
			writer.close();
		}

		catch (IOException e)
		{
			System.out.println("Error writing order to the file.");
			e.printStackTrace();
		}


	}
	public ArrayList<Index> readBayesNet(Domain domain)
	{
		BufferedReader domain_reader, var_reader;
		ArrayList<Index> subdomains = new ArrayList<Index>();

		try 
		{
			String class_path = System.getProperty("java.class.path").split(":")[0];
                        String[] domain_path = new String[] {class_path, "..", "tmp", "domain.txt"};
			String[] var_path = new String[] {class_path, "..", "tmp", "variables.txt"};
			domain_reader = new BufferedReader(new FileReader(String.join(File.separator, domain_path)));
			var_reader = new BufferedReader(new FileReader(String.join(File.separator, var_path)));
			String line;
			while ((line = domain_reader.readLine()) != null)
			{
				String[] node = line.split(",");
				String name = node[0];
				int num_of_state = Integer.parseInt(node[1].trim());
				domain.addDim(name, num_of_state);
				// add a variable and its number of states to domain
			}

			subdomains = new ArrayList<Index>();
			while ((line = var_reader.readLine()) != null)
			{
				String[] varNames = line.split(",");
				int[] varIDs = new int[varNames.length];
				for (int i = 0; i < varNames.length; i++)
					varIDs[i] = domain.index(varNames[i].trim());
				// create an Index object for each variable
				subdomains.add(new Index(domain, new IntSet(varIDs)));
			}

			domain_reader.close();
			var_reader.close();
		}

		catch (IOException e)
		{
			System.out.println("Error reading in the network.");
			e.printStackTrace();
		}

		return subdomains;

	}
}
