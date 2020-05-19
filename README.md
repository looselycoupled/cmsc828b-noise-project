* Overview

Alternate codebase for NMT project.

# DataSets

To create the datasets, run the following commands to add untranslated text to the source and target files.

```bash
cat baseline.tok.de untranslated_de_trg.05.tok.de >source.05.de
cat baseline.tok.de untranslated_de_trg.10.tok.de >source.10.de
cat baseline.tok.de untranslated_de_trg.20.tok.de >source.20.de
cat baseline.tok.de untranslated_de_trg.50.tok.de >source.50.de
cat baseline.tok.de untranslated_de_trg.100.tok.de >source.100.de

cat baseline.tok.en untranslated_de_trg.05.tok.de >target.05.en
cat baseline.tok.en untranslated_de_trg.10.tok.de >target.10.en
cat baseline.tok.en untranslated_de_trg.20.tok.de >target.20.en
cat baseline.tok.en untranslated_de_trg.50.tok.de >target.50.en
cat baseline.tok.en untranslated_de_trg.100.tok.de >target.100.en
```

To clean html entities out of the resulting files use:

```bash
cat target.05.en | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/target.05.en
cat target.10.en | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/target.10.en
cat target.20.en | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/target.20.en
cat target.50.en | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/target.50.en
cat target.100.en | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/target.100.en

cat source.05.de | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/source.05.de
cat source.10.de | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/source.10.de
cat source.20.de | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/source.20.de
cat source.50.de | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/source.50.de
cat source.100.de | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/source.100.de
```

For convenience, below are the commands to tar the files up before putting on s3.

```bash
tar czvf untranslated.05.tar.gz source.05.de target.05.en
tar czvf untranslated.10.tar.gz source.10.de target.10.en
tar czvf untranslated.20.tar.gz source.20.de target.20.en
tar czvf untranslated.50.tar.gz source.50.de target.50.en
tar czvf untranslated.100.tar.gz source.100.de target.100.en
```

To create the new datasets from 1% to 4% and then take the html entities out run the following commands.

```bash
head -n 35200  untranslated_de_trg.100.tok.de >untranslated_de_trg.01.tok.de
head -n 70400  untranslated_de_trg.100.tok.de >untranslated_de_trg.02.tok.de
head -n 105600  untranslated_de_trg.100.tok.de >untranslated_de_trg.03.tok.de
head -n 140800  untranslated_de_trg.100.tok.de >untranslated_de_trg.04.tok.de

cat baseline.tok.de untranslated_de_trg.01.tok.de >source.01.de
cat baseline.tok.de untranslated_de_trg.02.tok.de >source.02.de
cat baseline.tok.de untranslated_de_trg.03.tok.de >source.03.de
cat baseline.tok.de untranslated_de_trg.04.tok.de >source.04.de

cat baseline.tok.en untranslated_de_trg.01.tok.de >target.01.en
cat baseline.tok.en untranslated_de_trg.02.tok.de >target.02.en
cat baseline.tok.en untranslated_de_trg.03.tok.de >target.03.en
cat baseline.tok.en untranslated_de_trg.04.tok.de >target.04.en

cat target.01.en | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/target.01.en
cat target.02.en | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/target.02.en
cat target.03.en | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/target.03.en
cat target.04.en | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/target.04.en

cat source.01.de | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/source.01.de
cat source.02.de | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/source.02.de
cat source.03.de | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/source.03.de
cat source.04.de | perl -MHTML::Entities -pe 'decode_entities($_);' > ../wrangled/source.04.de

cd ../wrangled

tar czvf untranslated.01.tar.gz source.01.de target.01.en
tar czvf untranslated.02.tar.gz source.02.de target.02.en
tar czvf untranslated.03.tar.gz source.03.de target.03.en
tar czvf untranslated.04.tar.gz source.04.de target.04.en
```