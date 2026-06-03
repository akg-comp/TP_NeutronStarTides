using PDFIO

function extract_text(filename, outfilename)
    doc = pdDocOpen(filename)
    open(outfilename, "w") do f
        for i in 1:pdDocGetPageCount(doc)
            page = pdDocGetPage(doc, i)
            pdPageExtractText(f, page)
        end
    end
    pdDocClose(doc)
end

extract_text("Stability_NS_with_OoE_corrections (1).pdf", "stability_updated.txt")
