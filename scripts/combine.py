import pandas as pd
filelist = ['2000.csv', '2001.csv', '2002.csv', '2003.csv', '2004.csv', '2005.csv', '2006.csv', '2007.csv', '2008.csv', '2009.csv', '2010.csv', '2011.csv', '2012.csv', '2013.csv', '2014.csv']

def produceOneCSV(filelist, outfile):
   # Consolidate all CSV files into one object
   result_obj = pd.concat([pd.read_csv(file) for file in filelist])
   # Convert the above object into a csv file and export
   result_obj.to_csv(outfile, index=False, encoding="utf-8")

outfile = "APCombined.csv"
produceOneCSV(filelist, outfile)
