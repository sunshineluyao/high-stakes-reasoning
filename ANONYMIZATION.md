# Anonymous review export

This branch is designed to be exported as a history-free snapshot by an
anonymous repository service.

Before providing the review URL:

1. Export only the contents of this branch; do not expose `.git` history.
2. Confirm that the anonymous service masks the source owner and source URL.
3. Confirm that README links are repository-relative or point only to public
   third-party documentation.
4. Do not upload DAIC-WOZ participant IDs, labels, transcripts, facial features,
   case-level predictions, or model-generated rationales.
5. Replace the manuscript's artifact placeholder with the anonymous URL and
   inspect the compiled PDF metadata.
6. Restore author, citation, DOI, and permanent repository metadata only after
   the anonymous review period.
